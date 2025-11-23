import firebase_admin
from firebase_admin import credentials, messaging

try:
    CRED = credentials.Certificate("service-account.json")
    firebase_admin.initialize_app(CRED)
except ValueError:
    pass
except FileNotFoundError:
    print("FATAL: service-account.json not found. Cannot send notifications.")

def send_topic_notification(title, body, topic="notify_on_script"):
    """Sends a notification with dynamic title and body to a specific topic."""
    
    if not firebase_admin._apps:
        print("FCM ERROR: Firebase not initialized. Aborting notification.")
        return None

    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        topic=topic,
    )
    try:
        response = messaging.send(message)
        print(f"FCM INFO: Successfully sent message: {response}")
        return response
    except Exception as e:
        print(f"FCM ERROR: Failed to send notification: {e}")
        return None

if __name__ == "__main__":
    send_topic_notification(
        title="TEST ALERT", 
        body="This is a test notification from the standalone script.",
        topic="notify_on_script"
    )