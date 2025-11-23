import os
import asyncio
import logging
import threading
import numpy as np
import joblib
import warnings
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import skew, kurtosis
from sklearn.cross_decomposition import CCA
from pywizlight import wizlight, PilotBuilder 
from flask import Flask, request, jsonify

try:
    from send_notification import send_topic_notification
except ImportError:
    print("FATAL: Could not import send_topic_notification. Ensure send_notification.py is present.")
    exit()

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("BCI_SERVER")

BULB_IP = "10.52.62.70"
MODEL_PATH = "improved_errp_classifier.pkl"
TARGET_FREQ = 7.5
MONITOR_FREQ = 10.0
ERROR_CLASS = 1

UPLOAD_FOLDER = 'uploaded_eeg_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== 1. SIMPLE WIZ BULB CONTROL CLASS ====

class SimpleWiz:
    def __init__(self, ip):
        self.ip = ip
        self.bulb = wizlight(ip)
        self.state = False

    async def get_state(self):
        try:
            state = await self.bulb.updateState()
            self.state = state.get_state()
            return self.state
        except Exception as e:
            LOG.error(f"Could not get bulb state. Defaulting internally to OFF. Error: {e}")
            self.state = False
            return self.state

    async def turn_on(self):
        await self.bulb.turn_on(PilotBuilder(brightness=255, colortemp=4000))
        LOG.info("💡 Bulb ON")
        self.state = True

    async def turn_off(self):
        await self.bulb.turn_off()
        LOG.info("💡 Bulb OFF")
        self.state = False

    async def toggle(self):
        if self.state:
            await self.turn_off()
        else:
            await self.turn_on()

# ==== 2. SSVEP CLASSIFIER CLASS (FBCCA) ====

class StateOfArtSSVEPClassifier:
    def __init__(self, sampling_rate=500, frequencies=[TARGET_FREQ, MONITOR_FREQ], num_harmonics=5, num_subbands=5):
        self.fs = sampling_rate
        self.frequencies = frequencies
        self.num_harmonics = num_harmonics
        self.num_subbands = num_subbands
        self.signal_duration = 10
        self.occipital_channels = [3,4,5,6,7]
        self.filterbanks = [(5 + i*4, min(40 + i*4, self.fs/2 -1)) for i in range(self.num_subbands)]
        self.fb_weights = np.array([n**(-1.25)+0.25 for n in range(1, self.num_subbands+1)])
        LOG.info(f"✓ Initialized: FBCCA | freqs: {self.frequencies}")

    def load_data(self, filepath):
        try:
            return np.loadtxt(filepath)[:, :8]
        except Exception as e:
            LOG.error(f"Error reading {filepath}: {e}")
            return None

    def preprocess(self, eeg_data):
        cleaned = np.copy(eeg_data)
        nyq = 0.5 * self.fs
        b, a = signal.butter(6, [3/nyq, 45/nyq], btype="band")
        filtered = signal.filtfilt(b, a, cleaned, axis=0)
        b_notch, a_notch = signal.iirnotch(50, 30, self.fs)
        filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=0)
        return (filtered - np.mean(filtered, axis=0)) / (np.std(filtered, axis=0) + 1e-10)

    def apply_filterbank(self, eeg_data, band_idx):
        low, high = self.filterbanks[band_idx]
        nyq = 0.5 * self.fs
        b, a = signal.cheby2(4, 40, [low/nyq, high/nyq], btype="band")
        return signal.filtfilt(b, a, eeg_data, axis=0)

    def generate_reference(self, freq, n_samples):
        t = np.arange(n_samples) / self.fs
        ref = []
        for h in range(1, self.num_harmonics+1):
            ref += [np.sin(2*np.pi*h*freq*t), np.cos(2*np.pi*h*freq*t)]
        return np.array(ref).T

    def cca_correlate(self, X, Y):
        try:
            cca = CCA(n_components=1)
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
            return abs(np.corrcoef(X_c[:,0], Y_c[:,0])[0,1])
        except Exception:
            return 0.0

    def fbcca_classify(self, X_test):
        fb_scores = {f: [] for f in self.frequencies}
        for band_idx in range(self.num_subbands):
            Xf = self.apply_filterbank(X_test, band_idx)
            for f in self.frequencies:
                ref = self.generate_reference(f, Xf.shape[0])
                fb_scores[f].append(self.cca_correlate(Xf, ref))
        
        final_scores = {f: np.sum(np.array(fb_scores[f]) * self.fb_weights) for f in self.frequencies}
        pred_freq = max(final_scores, key=final_scores.get)
        return pred_freq, final_scores

    def predict_file(self, filepath):
        eeg = self.load_data(filepath)
        if eeg is None:
            return None, {}
        eeg_proc = self.preprocess(eeg)
        ssvep = eeg_proc[:int(self.signal_duration*self.fs), self.occipital_channels]
        return self.fbcca_classify(ssvep)

# ==== 3. ErrP TESTER CLASS (Prediction Engine) ====

class ErrPTester:
    def __init__(self, model_path=MODEL_PATH):
        LOG.info(f"Loading ErrP Model: {model_path}")
        try:
            model_data = joblib.load(model_path)
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data.get('feature_selector', None)
            self.sampling_rate = model_data['sampling_rate']
            self.use_advanced_features = model_data.get('use_advanced_features', True)
            self.n_features_expected = self.scaler.n_features_in_
            LOG.info(f"✓ ErrP Model loaded. Expected features: {self.n_features_expected}")
        except Exception as e:
            LOG.error(f"❌ ERROR loading model: {str(e)}")
            raise

        self.channel_names = ['Fz', 'Cz', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4']

    def load_data(self, filepath):
        data = np.loadtxt(filepath)
        eeg_data = data[:, :8] / 1000.0
        return eeg_data, data[:, 8:11], data[:, 11], data[:, 12]

    def remove_artifacts(self, data, threshold_std=3.5):
        cleaned_data = data.copy()
        for ch in range(data.shape[1]):
            mean_val = np.mean(data[:, ch])
            std_val = np.std(data[:, ch])
            upper = mean_val + threshold_std * std_val
            lower = mean_val - threshold_std * std_val
            cleaned_data[:, ch] = np.clip(cleaned_data[:, ch], lower, upper)
        return cleaned_data

    def adaptive_filter(self, data):
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            v = np.var(data[:, ch])
            lowcut, highcut = (0.5, 8.0) if v > 100 else (0.1, 10.0)
            nyq = 0.5 * self.sampling_rate
            b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
            filtered[:, ch] = filtfilt(b, a, data[:, ch])
        return filtered

    def extract_errp_epoch(self, eeg_data, start=12.0, dur=3.0):
        s = int(start * self.sampling_rate)
        d = int(dur * self.sampling_rate)
        e = min(len(eeg_data), s + d)
        return eeg_data[s:e, :]

    def compute_power_spectral_density(self, signal_data, ch):
        freqs, psd = welch(signal_data[:, ch], fs=self.sampling_rate, nperseg=min(256, len(signal_data)))
        delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
        theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
        alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
        return delta, theta, alpha

    def extract_wavelet_features(self, signal_data, ch):
        scales = np.arange(1, 32, 2)
        coeff = signal.cwt(signal_data[:, ch], signal.ricker, scales)
        return np.mean(np.abs(coeff)), np.std(coeff), np.sum(coeff**2)

    def advanced_baseline_correction(self, epoch, method='linear'):
        corrected = np.zeros_like(epoch)
        for ch in range(epoch.shape[1]):
            x = np.arange(len(epoch))
            if method == 'linear':
                coeffs = np.polyfit(x, epoch[:, ch], 1)
                trend = np.polyval(coeffs, x)
                corrected[:, ch] = epoch[:, ch] - trend
            else:
                baseline = np.mean(epoch[:100, ch])
                corrected[:, ch] = epoch[:, ch] - baseline
        return corrected

    def extract_comprehensive_features(self, epoch):
        features = []
        epoch = self.advanced_baseline_correction(epoch, method='linear')
        expected = int(3 * self.sampling_rate)
        if len(epoch) < expected:
            pad = expected - len(epoch)
            epoch = np.vstack([epoch, np.zeros((pad, epoch.shape[1]))])

        be = int(0.2 * self.sampling_rate)
        ps = int(0.2 * self.sampling_rate)
        pe = int(0.4 * self.sampling_rate)
        ls = int(0.4 * self.sampling_rate)
        le = int(0.6 * self.sampling_rate)
        xs = int(0.6 * self.sampling_rate)
        pr = [0,1,2]

        for ch in pr:
            d = epoch[:, ch]
            features.extend([
                np.mean(d), np.std(d), np.var(d), np.ptp(d),
                np.max(d), np.min(d), skew(d), kurtosis(d),
                np.percentile(d,25), np.percentile(d,50), np.percentile(d,75)
            ])

        if pe > ps:
            base = epoch[:be, :]
            pew = epoch[ps:pe, :]
            for ch in pr:
                bm = np.mean(base[:, ch])
                features.append(bm)
                pw = pew[:, ch]
                pmean = np.mean(pw)
                features.extend([
                    pmean, np.min(pw), np.max(pw),
                    np.std(pw), np.ptp(pw),
                    pmean - bm,
                    np.argmin(pw)/len(pw), np.argmax(pw)/len(pw),
                    (pw[-1]-pw[0])/len(pw)
                ])

            if le > ls:
                lw = epoch[ls:le, :]
                for ch in pr:
                    lm, ls_ = np.mean(lw[:, ch]), np.std(lw[:, ch])
                    features.extend([lm, ls_, lm - np.mean(epoch[ps:pe, ch])])

            if xs < len(epoch):
                xw = epoch[xs:, :]
                for ch in pr:
                    features.extend([np.mean(xw[:, ch]), np.std(xw[:, ch])])

        if pe > ps:
            pew = epoch[ps:pe, :]
            la = np.mean(pew[:,4]) + np.mean(pew[:,3])
            ra = np.mean(pew[:,6]) + np.mean(pew[:,7])
            features.append(la - ra)
            features.append(np.mean(pew[:,0]) - np.mean(pew[:,2]))
            features.extend([
                np.mean(pew[:,0]) - np.mean(pew[:,1]),
                np.mean(pew[:,1]) - np.mean(pew[:,2])
            ])
            gfp = np.std(pew, axis=1)
            features.extend([np.mean(gfp), np.max(gfp), np.std(gfp)])

        if self.use_advanced_features:
            for ch in pr:
                try:
                    delta, theta, alpha = self.compute_power_spectral_density(epoch, ch)
                    total = delta + theta + alpha + 1e-10
                    features.extend([
                        delta, theta, alpha,
                        delta/total, theta/total, alpha/total,
                        theta/(alpha+1e-10)
                    ])
                except:
                    features.extend([0]*7)

            for ch in pr:
                try:
                    wm, ws, we = self.extract_wavelet_features(epoch, ch)
                    features.extend([wm, ws, we])
                except:
                    features.extend([0,0,0])

        for ch in pr:
            if len(epoch) > 1:
                diff = np.diff(epoch[:, ch])
                features.extend([np.mean(np.abs(diff)), np.std(diff), np.max(np.abs(diff))])
            else:
                features.extend([0,0,0])

        if pe > ps:
            pew = epoch[ps:pe, :]
            if len(pew) > 2:
                try:
                    features.append(np.corrcoef(pew[:,0], pew[:,1])[0,1])
                    features.append(np.corrcoef(pew[:,1], pew[:,2])[0,1])
                except:
                    features.extend([0,0])
            else:
                features.extend([0,0])

        return np.array(features)

    def predict_from_file(self, filepath, verbose=True):
        try:
            eeg,_,_,_ = self.load_data(filepath)
            eeg = self.remove_artifacts(eeg)
            filtered = self.adaptive_filter(eeg)
            epoch = self.extract_errp_epoch(filtered)
            features = self.extract_comprehensive_features(epoch)

            if len(features) != self.n_features_expected:
                LOG.error(f"Feature mismatch: {len(features)} vs {self.n_features_expected}")
                return None, None

            fs = self.scaler.transform(features.reshape(1,-1))
            if self.feature_selector is not None:
                fs = self.feature_selector.transform(fs)

            pred = self.classifier.predict(fs)[0]
            prob = self.classifier.predict_proba(fs)[0]
            return pred, prob

        except Exception as e:
            LOG.error(f"❌ ErrP prediction error: {e}")
            return None, None

# ==== 4. BCI MASTER CONTROL LOGIC (The integrated sequencer) ====

async def process_eeg_file_and_control_bulb(filepath):
    LOG.info(f"\n--- STARTING BCI ANALYSIS for {os.path.basename(filepath)} ---")

    ssvep_success = False
    ssvep_monitor = False

    try:
        wiz = SimpleWiz(BULB_IP)
        S_initial = await wiz.get_state()
        LOG.info(f"Initial Bulb State: {'ON' if S_initial else 'OFF'}")

        ssvep_clf = StateOfArtSSVEPClassifier(frequencies=[TARGET_FREQ, MONITOR_FREQ])
        errp = ErrPTester(model_path=MODEL_PATH)
    except Exception as e:
        LOG.error(f"Initialization Failed: {e}")
        return

    name = os.path.basename(filepath)
    if name.startswith("7.5"):
        intended = TARGET_FREQ
    elif name.startswith("10"):
        intended = MONITOR_FREQ
    else:
        intended = None

    pred_freq, _ = ssvep_clf.predict_file(filepath)
    S_SSVEP = S_initial
    mis = False

    if pred_freq == TARGET_FREQ:
        if intended == TARGET_FREQ:
            await wiz.toggle()
            S_SSVEP = wiz.state
            ssvep_success = True
            LOG.info(f"Detected {TARGET_FREQ} Hz. Bulb Toggled.")
        else:
            mis = True
            ssvep_monitor = True
            LOG.warning("MISCLASSIFIED TARGET: Expected 10, detected 7.5")
            S_SSVEP = S_initial

    elif pred_freq == MONITOR_FREQ:
        if intended == MONITOR_FREQ:
            ssvep_monitor = True
            LOG.info("Detected 10 Hz Monitor.")
        else:
            mis = True
            ssvep_success = True
            LOG.warning("MISCLASSIFIED TARGET: Expected 7.5, detected 10")
        S_SSVEP = S_initial

    pred, prob = errp.predict_from_file(filepath, verbose=False)
    conf_err = prob[ERROR_CLASS]

    if pred == ERROR_CLASS:
        LOG.warning("ErrP DETECTED")
        if mis:
            send_topic_notification(
                title="DIAGNOSTIC ERROR (SSVEP MISMATCH)",
                body="Critical: SSVEP predicted the wrong target. Action REVERTED."
            )
        elif intended == TARGET_FREQ:
            send_topic_notification(
                title="REVERSION SUCCESS (7.5 Hz Trial Failed)",
                body="ErrP detected! Action reversed."
            )
        elif intended == MONITOR_FREQ:
            send_topic_notification(
                title="CRITICAL MONITORING ALERT (10 Hz Error)",
                body="ErrP detected during monitor. Alert."
            )
        else:
            send_topic_notification(
                title="FALSE ALARM / AMBIGUOUS ERROR",
                body="ErrP detected during unknown trial."
            )

        if S_SSVEP != S_initial:
            await wiz.toggle()
            LOG.info("Reverted bulb toggle.")
        else:
            LOG.info("No reversion required.")

    else:
        LOG.info("ErrP NOT DETECTED")
        if mis:
            LOG.warning("Misclassification occurred but stable (No ErrP).")
        elif intended == TARGET_FREQ:
            send_topic_notification(
                title="Bulb Toggled",
                body=f"Target achieved. Bulb now {'ON' if S_SSVEP else 'OFF'}."
            )
        elif intended == MONITOR_FREQ:
            send_topic_notification(
                title="Patient Calls for Help",
                body="10 Hz detected. Assistance needed."
            )

    LOG.info(f"FINAL BULB STATE: {'ON' if wiz.state else 'OFF'}")

    try:
        os.remove(filepath)
    except:
        pass

# ==== 5. NETWORK SERVER (FLASK) SETUP ====

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message':'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'message':'No selected file'}), 400
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)

    t = threading.Thread(target=run_bci_async, args=(path,))
    t.start()

    return jsonify({'message':f'File received and processing started for {f.filename}'}), 200

def run_bci_async(filepath):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if 'cwt' not in signal.__dict__ and hasattr(signal, 'cwt'):
            signal.cwt = signal.cwt
        if 'ricker' not in signal.__dict__ and hasattr(signal, 'ricker'):
            signal.ricker = signal.ricker
        loop.run_until_complete(process_eeg_file_and_control_bulb(filepath))
    except Exception as e:
        LOG.error(f"Error during BCI processing: {e}")
    finally:
        loop.close()

if __name__ == '__main__':
    LOG.info("Starting BCI Server...")
    LOG.info("Listening on /upload")
    app.run(host='0.0.0.0', port=5000, debug=False)
