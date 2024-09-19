import random
import uuid
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from faker import Faker
from pymongo import MongoClient

# Initialize Faker
fake = Faker()

# MongoDB connection setup
# Replace the URI with your actual MongoDB connection string
mongo_client = MongoClient("mongodb://softgen:QWAmTnsdBUaTL2z@118.139.167.125:27017/")
database = mongo_client["Summary"]
collection = database["OutstandingSummary"]

def generate_random_document():
    client_id = str(uuid.uuid4())
    ledger_name = fake.company()
    total_transactions = random.randint(1, 10)
    total_delayed = random.randint(0, total_transactions)
    delay_percentage = (total_delayed / total_transactions) * 100 if total_transactions > 0 else 0
    amount_due = random.randint(1000, 500000)
    average_delay_days = random.randint(1, 180)
    last_action = random.choice(['Send Reminder', 'Check Payment Status', 'Contact Client'])
    last_outcome = random.choice(['Payment Pending', 'Payment Received', 'No Response'])
    action_history = []

    # Generate random action history
    num_actions = random.randint(1, 5)
    for _ in range(num_actions):
        action_date = datetime.now() - timedelta(days=random.randint(0, 365))
        action_entry = {
            'action': random.choice(['Send Reminder', 'Check Payment Status', 'Contact Client']),
            'outcome': random.choice(['Pending', 'Completed', 'Failed']),
            'date': action_date
        }
        action_history.append(action_entry)

    document = {
        '_id': ObjectId(),
        'client_id': client_id,
        'ledger_name': ledger_name,
        'total_transactions': total_transactions,
        'total_delayed': total_delayed,
        'delay_percentage': delay_percentage,
        'amount_due': amount_due,
        'average_delay_days': average_delay_days,
        'last_action': last_action,
        'last_outcome': last_outcome,
        'action_history': action_history
    }

    return document

def main():
    try:
        num_records = int(input("Enter the number of records to generate: "))
        documents = []
        for _ in range(num_records):
            doc = generate_random_document()
            documents.append(doc)

        # Insert documents into MongoDB
        result = collection.insert_many(documents)
        print(f"Inserted {len(result.inserted_ids)} documents into MongoDB collection '{collection.name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        mongo_client.close()

if __name__ == "__main__":
    main()

