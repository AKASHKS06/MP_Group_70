import numpy as np
import os
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ImprovedErrPClassifier:
    def __init__(self, sampling_rate=500, use_advanced_features=True):
        self.sampling_rate = sampling_rate
        self.scaler = RobustScaler()
        self.classifier = None
        self.channel_names = ['Fz', 'Cz', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4']
        self.eeg_units = 'nV'
        self.use_advanced_features = use_advanced_features
        self.feature_selector = None

    def load_data(self, filepath):
        data = np.loadtxt(filepath)
        eeg_data = data[:, :8] / 1000.0
        accelerometer = data[:, 8:11]
        triggers = data[:, 11]
        timestamps = data[:, 12]
        return eeg_data, accelerometer, triggers, timestamps

    def adaptive_filter(self, data):
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            signal_var = np.var(data[:, ch])
            if signal_var > 100:
                lowcut, highcut = 0.5, 8.0
            else:
                lowcut, highcut = 0.1, 10.0
            nyquist = 0.5 * self.sampling_rate
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
        return filtered_data

    def remove_artifacts(self, data, threshold_std=3.5):
        cleaned_data = data.copy()
        for ch in range(data.shape[1]):
            mean_val = np.mean(data[:, ch])
            std_val = np.std(data[:, ch])
            upper_bound = mean_val + threshold_std * std_val
            lower_bound = mean_val - threshold_std * std_val
            cleaned_data[:, ch] = np.clip(cleaned_data[:, ch], lower_bound, upper_bound)
        return cleaned_data

    def advanced_baseline_correction(self, epoch, method='mean'):
        corrected = np.zeros_like(epoch)
        for ch in range(epoch.shape[1]):
            if method == 'mean':
                baseline = np.mean(epoch[:100, ch])
                corrected[:, ch] = epoch[:, ch] - baseline
            elif method == 'linear':
                x = np.arange(len(epoch))
                coeffs = np.polyfit(x, epoch[:, ch], 1)
                trend = np.polyval(coeffs, x)
                corrected[:, ch] = epoch[:, ch] - trend
            elif method == 'polynomial':
                x = np.arange(len(epoch))
                coeffs = np.polyfit(x, epoch[:, ch], 2)
                trend = np.polyval(coeffs, x)
                corrected[:, ch] = epoch[:, ch] - trend
        return corrected

    def extract_errp_epoch(self, eeg_data, errp_window_start=12.0, errp_window_duration=3.0):
        start_sample = int(errp_window_start * self.sampling_rate)
        duration_samples = int(errp_window_duration * self.sampling_rate)
        if start_sample + duration_samples > len(eeg_data):
            end_sample = len(eeg_data)
        else:
            end_sample = start_sample + duration_samples
        epoch = eeg_data[start_sample:end_sample, :]
        return epoch

    def compute_power_spectral_density(self, signal_data, channel_idx):
        freqs, psd = welch(signal_data[:, channel_idx], fs=self.sampling_rate, nperseg=min(256, len(signal_data)))
        delta_power = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
        theta_power = np.mean(psd[(freqs >= 4) & (freqs < 8)])
        alpha_power = np.mean(psd[(freqs >= 8) & (freqs < 13)])
        return delta_power, theta_power, alpha_power

    def extract_wavelet_features(self, signal_data, channel_idx):
        from scipy.signal import cwt, ricker
        scales = np.arange(1, 32, 2)
        coefficients = cwt(signal_data[:, channel_idx], ricker, scales)
        wavelet_mean = np.mean(np.abs(coefficients))
        wavelet_std = np.std(coefficients)
        wavelet_energy = np.sum(coefficients**2)
        return wavelet_mean, wavelet_std, wavelet_energy

    def extract_comprehensive_features(self, epoch):
        features = []
        epoch_corrected = self.advanced_baseline_correction(epoch, method='linear')
        baseline_end = int(0.2 * self.sampling_rate)
        pe_start = int(0.2 * self.sampling_rate)
        pe_end = int(0.4 * self.sampling_rate)
        late_start = int(0.4 * self.sampling_rate)
        late_end = int(0.6 * self.sampling_rate)
        extended_start = int(0.6 * self.sampling_rate)
        expected_samples = int(3.0 * self.sampling_rate)
        if len(epoch_corrected) < expected_samples:
            print(f"Warning: Expected {expected_samples} samples but got {len(epoch_corrected)}")
        if len(epoch_corrected) < pe_end:
            pe_end = len(epoch_corrected)
        if len(epoch_corrected) < late_end:
            late_end = len(epoch_corrected)
        priority_channels = [0, 1, 2]

        for ch in priority_channels:
            channel_data = epoch_corrected[:, ch]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                np.ptp(channel_data),
                np.max(channel_data),
                np.min(channel_data),
                skew(channel_data),
                kurtosis(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 50),
                np.percentile(channel_data, 75)
            ])

        if pe_end > pe_start and len(epoch_corrected) >= pe_end:
            baseline_window = epoch_corrected[:baseline_end, :]
            pe_window = epoch_corrected[pe_start:pe_end, :]
            for ch in priority_channels:
                baseline_mean = np.mean(baseline_window[:, ch])
                pe_mean = np.mean(pe_window[:, ch])
                pe_min = np.min(pe_window[:, ch])
                pe_max = np.max(pe_window[:, ch])
                pe_std = np.std(pe_window[:, ch])
                pe_ptp = np.ptp(pe_window[:, ch])
                features.extend([
                    baseline_mean,
                    pe_mean,
                    pe_min,
                    pe_max,
                    pe_std,
                    pe_ptp,
                    pe_mean - baseline_mean
                ])
                min_idx = np.argmin(pe_window[:, ch])
                max_idx = np.argmax(pe_window[:, ch])
                features.extend([
                    min_idx / len(pe_window),
                    max_idx / len(pe_window)
                ])
                pe_slope = (pe_window[-1, ch] - pe_window[0, ch]) / len(pe_window)
                features.append(pe_slope)

            if late_end > late_start and late_end <= len(epoch_corrected):
                late_window = epoch_corrected[late_start:late_end, :]
                for ch in priority_channels:
                    late_mean = np.mean(late_window[:, ch])
                    late_std = np.std(late_window[:, ch])
                    features.extend([late_mean, late_std, late_mean - pe_mean])

            if extended_start < len(epoch_corrected):
                extended_window = epoch_corrected[extended_start:, :]
                for ch in priority_channels:
                    features.extend([
                        np.mean(extended_window[:, ch]),
                        np.std(extended_window[:, ch])
                    ])

        if pe_end > pe_start and len(epoch_corrected) >= pe_end:
            pe_window = epoch_corrected[pe_start:pe_end, :]
            left_activity = np.mean(pe_window[:, 4]) + np.mean(pe_window[:, 3])
            right_activity = np.mean(pe_window[:, 6]) + np.mean(pe_window[:, 7])
            features.append(left_activity - right_activity)
            frontal = np.mean(pe_window[:, 0])
            parietal = np.mean(pe_window[:, 2])
            features.append(frontal - parietal)
            features.extend([
                np.mean(pe_window[:, 0]) - np.mean(pe_window[:, 1]),
                np.mean(pe_window[:, 1]) - np.mean(pe_window[:, 2])
            ])
            gfp = np.std(pe_window, axis=1)
            features.extend([
                np.mean(gfp),
                np.max(gfp),
                np.std(gfp)
            ])

        if self.use_advanced_features and len(epoch_corrected) > 256:
            for ch in priority_channels:
                try:
                    delta, theta, alpha = self.compute_power_spectral_density(epoch_corrected, ch)
                    features.extend([delta, theta, alpha])
                    total_power = delta + theta + alpha + 1e-10
                    features.extend([
                        delta / total_power,
                        theta / total_power,
                        alpha / total_power
                    ])
                    features.append(theta / (alpha + 1e-10))
                except:
                    features.extend([0] * 7)

        if self.use_advanced_features and len(epoch_corrected) > 100:
            for ch in priority_channels:
                try:
                    w_mean, w_std, w_energy = self.extract_wavelet_features(epoch_corrected, ch)
                    features.extend([w_mean, w_std, w_energy])
                except:
                    features.extend([0, 0, 0])

        for ch in priority_channels:
            if len(epoch_corrected) > 1:
                diff = np.diff(epoch_corrected[:, ch])
                features.extend([
                    np.mean(np.abs(diff)),
                    np.std(diff),
                    np.max(np.abs(diff))
                ])

        if pe_end > pe_start and len(epoch_corrected) >= pe_end:
            pe_window = epoch_corrected[pe_start:pe_end, :]
            if len(pe_window) > 2:
                try:
                    fz_cz_corr = np.corrcoef(pe_window[:, 0], pe_window[:, 1])[0, 1]
                    cz_pz_corr = np.corrcoef(pe_window[:, 1], pe_window[:, 2])[0, 1]
                    features.extend([fz_cz_corr, cz_pz_corr])
                except:
                    features.extend([0, 0])
        return np.array(features)

    def load_dataset(self, data_directory, augment_data=False):
        X = []
        y = []
        filenames = []
        for filename in os.listdir(data_directory):
            if filename.endswith('.easy'):
                filepath = os.path.join(data_directory, filename)
                parts = filename.replace('.easy', '').split('_')
                label_indicator = parts[-1]
                if label_indicator.lower() == 'r':
                    label = 0
                elif label_indicator.lower() == 'w':
                    label = 1
                else:
                    continue
                try:
                    eeg_data, _, _, _ = self.load_data(filepath)
                    eeg_data = self.remove_artifacts(eeg_data)
                    filtered_data = self.adaptive_filter(eeg_data)
                    epoch = self.extract_errp_epoch(filtered_data)
                    features = self.extract_comprehensive_features(epoch)
                    X.append(features)
                    y.append(label)
                    filenames.append(filename)
                    if augment_data:
                        for _ in range(2):
                            noise = np.random.normal(0, 0.5, eeg_data.shape)
                            augmented_eeg = eeg_data + noise
                            augmented_filtered = self.adaptive_filter(augmented_eeg)
                            augmented_epoch = self.extract_errp_epoch(augmented_filtered)
                            augmented_features = self.extract_comprehensive_features(augmented_epoch)
                            X.append(augmented_features)
                            y.append(label)
                            filenames.append(f"{filename}_aug")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
        X = np.array(X)
        y = np.array(y)
        print(f"\n{'='*60}")
        print(f"Dataset loaded successfully!")
        print(f"{'='*60}")
        print(f"Total samples: {len(X)}")
        print(f"  - No Error/Correct (r): {np.sum(y == 0)} samples")
        print(f"  - Error/Wrong (w): {np.sum(y == 1)} samples")
        print(f"Feature vector dimension: {X.shape[1] if len(X) > 0 else 0}")
        print(f"Class balance: {np.sum(y == 1) / len(y) * 100:.1f}% error samples")
        unique, counts = np.unique(y, return_counts=True)
        imbalance_ratio = np.max(counts) / np.min(counts)
        if imbalance_ratio > 1.5:
            print(f"⚠️  Warning: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
            print(f"   Consider using SMOTE or class weighting")
        print(f"{'='*60}\n")
        return X, y, filenames

    def train(self, X, y, classifier_type='Ensemble', test_size=0.2,
              random_state=42, use_smote=True, use_feature_selection=True,
              hyperparameter_tuning=False):
        if len(X) < 10:
            print("⚠️  Warning: Very small dataset. Results may not be reliable.")
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            print("❌ Error: Only one class present in dataset.")
            return None
        print(f"\n{'='*60}")
        print("TRAINING IMPROVED ErrP CLASSIFIER")
        print(f"{'='*60}\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples\n")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        if use_smote and len(X_train) >= 10:
            try:
                print("Applying SMOTE for class balancing...")
                smote = SMOTETomek(random_state=random_state)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                print(f"After SMOTE: {len(X_train_scaled)} samples\n")
            except Exception as e:
                print(f"SMOTE failed: {e}. Continuing without SMOTE.\n")
        if use_feature_selection and X_train_scaled.shape[1] > 30:
            print("Performing feature selection...")
            n_features = min(50, X_train_scaled.shape[1])
            self.feature_selector = SelectKBest(f_classif, k=n_features)
            X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_scaled = self.feature_selector.transform(X_test_scaled)
            print(f"Selected {n_features} best features\n")
        if classifier_type == 'LDA':
            self.classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            print("Training: Linear Discriminant Analysis (LDA)")
        elif classifier_type == 'SVM':
            if hyperparameter_tuning and len(X_train_scaled) > 20:
                print("Training: SVM with hyperparameter tuning...")
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf', 'linear']
                }
                svm = SVC(probability=True, random_state=random_state)
                self.classifier = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            else:
                print("Training: Support Vector Machine (SVM)")
                self.classifier = SVC(kernel='rbf', C=10, gamma='scale',
                                     probability=True, random_state=random_state,
                                     class_weight='balanced')
        elif classifier_type == 'XGBoost':
            print("Training: XGBoost Classifier")
            scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
            self.classifier = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state
            )
        elif classifier_type == 'Ensemble':
            print("Training: Ensemble of classifiers (LDA + SVM + XGBoost)")
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                     random_state=random_state, class_weight='balanced')
            xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                               random_state=random_state)
            self.classifier = VotingClassifier(
                estimators=[('lda', lda), ('svm', svm), ('xgb', xgb)],
                voting='soft'
            )
        else:
            raise ValueError("classifier_type must be 'LDA', 'SVM', 'XGBoost', or 'Ensemble'")
        self.classifier.fit(X_train_scaled, y_train)
        y_pred = self.classifier.predict(X_test_scaled)
        y_pred_proba = self.classifier.predict_proba(X_test_scaled)[:, 1]
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}\n")
        print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC-ROC Score: {auc:.3f}")
        except:
            pass
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['No Error (r)', 'Error (w)'],
                                   zero_division=0))
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("\nTrue Negatives:", cm[0,0] if cm.shape[0] > 0 else 0)
        print("False Positives:", cm[0,1] if cm.shape[0] > 1 else 0)
        print("False Negatives:", cm[1,0] if cm.shape[0] > 1 else 0)
        print("True Positives:", cm[1,1] if cm.shape[0] > 1 else 0)
        if len(X_train_scaled) >= 5:
            n_folds = min(5, np.min(np.unique(y_train, return_counts=True)[1]))
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(self.classifier, X_train_scaled, y_train,
                                       cv=cv, scoring='accuracy')
            print(f"\n{n_folds}-Fold Cross-Validation:")
            print(f"Mean Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"{'='*60}\n")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def predict(self, eeg_data, return_confidence=True):
        if self.classifier is None:
            raise ValueError("Model not trained.")
        if isinstance(eeg_data, str):
            eeg_data, _, _, _ = self.load_data(eeg_data)
        eeg_data = self.remove_artifacts(eeg_data)
        filtered_data = self.adaptive_filter(eeg_data)
        epoch = self.extract_errp_epoch(filtered_data)
        features = self.extract_comprehensive_features(epoch)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        if self.feature_selector is not None:
            features_scaled = self.feature_selector.transform(features_scaled)
        prediction = self.classifier.predict(features_scaled)[0]
        if return_confidence and hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            return prediction, probabilities
        else:
            return prediction

    def save_model(self, filepath='improved_errp_classifier.pkl'):
        if self.classifier is None:
            raise ValueError("No trained model to save.")
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'sampling_rate': self.sampling_rate,
            'use_advanced_features': self.use_advanced_features
        }
        joblib.dump(model_data, filepath)
        print(f"\n✅ Model saved to {filepath}")

    def load_model(self, filepath='improved_errp_classifier.pkl'):
        model_data = joblib.load(filepath)
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data.get('feature_selector', None)
        self.sampling_rate = model_data['sampling_rate']
        self.use_advanced_features = model_data.get('use_advanced_features', True)
        print(f"\n✅ Model loaded from {filepath}")


def main():
        print("\n" + "="*60)
        print("IMPROVED ErrP CLASSIFIER - OPTIMIZED FOR YOUR DATA")
        print("="*60 + "\n")
        errp_clf = ImprovedErrPClassifier(
            sampling_rate=500,
            use_advanced_features=True
        )
        data_directory = "./eeg_data"
        print("Loading data with augmentation to increase sample size...")
        X, y, filenames = errp_clf.load_dataset(
            data_directory,
            augment_data=True
        )
        if len(X) > 0:
            print("\nTraining with FULL optimization pipeline:")
            print("✓ Ensemble classifier (LDA + SVM + XGBoost)")
            print("✓ SMOTE for class balancing")
            print("✓ Feature selection for best features")
            print("✓ Adaptive filtering & artifact removal")
            print("✓ Advanced baseline correction\n")
            errp_clf.train(
                X, y,
                classifier_type='Ensemble',
                use_smote=True,
                use_feature_selection=True,
                hyperparameter_tuning=False
            )
            errp_clf.save_model('improved_errp_classifier.pkl')
            print("\n" + "="*60)
            print("TESTING PREDICTION")
            print("="*60)
            if len(filenames) > 0:
                test_file = None
                for fname in filenames:
                    if not fname.endswith('_aug'):
                        test_file = fname
                        break
                if test_file:
                    test_filepath = os.path.join(data_directory, test_file)
                    print(f"Testing on: {test_file}")
                    try:
                        prediction, probabilities = errp_clf.predict(test_filepath)
                        print(f"\nPrediction: {'⚠️ ERROR DETECTED (w)' if prediction == 1 else '✓ NO ERROR (r)'}")
                        print(f"Confidence:")
                        print(f"  - No Error: {probabilities[0]:.1%}")
                        print(f"  - Error:    {probabilities[1]:.1%}")
                        if probabilities[1] > 0.7:
                            print("\n→ High confidence ERROR detection - would trigger revert")
                        elif probabilities[1] > 0.5:
                            print("\n→ Moderate confidence ERROR - might trigger revert")
                        else:
                            print("\n→ No significant error detected")
                    except Exception as e:
                        print(f"Prediction error: {e}")
            print("\n" + "="*60)
            print("NEXT STEPS TO IMPROVE FURTHER")
            print("="*60)
            print("1. 📊 Collect more data:")
            print("   - Current: ~62 samples")
            print("   - Target: 100+ samples (50+ per class)")
            print("   - More data = better accuracy!")
            print()
            print("2. ✅ Verify ErrP presence:")
            print("   - Make sure feedback is CLEAR at 12s")
            print("   - Subject must NOTICE when wrong")
            print("   - Visual: Use ErrP Signal Visualizer")
            print()
            print("3. 🔧 Try hyperparameter tuning:")
            print("   - Set hyperparameter_tuning=True")
            print("   - Takes longer but optimizes classifier")
            print()
            print("4. 📈 Balance your classes:")
            print("   - Try to collect equal 'r' and 'w' samples")
            print("   - Currently: 25 'r' vs 37 'w'")
            print()
            print("5. 🎯 Expected accuracy with good data:")
            print("   - Fair: 60-70%")
            print("   - Good: 70-80%")
            print("   - Excellent: 80-90%")
            print("="*60 + "\n")


if __name__ == "__main__":
    main()
