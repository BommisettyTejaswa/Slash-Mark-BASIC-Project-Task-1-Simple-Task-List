import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

# Initialize an empty task list
tasks = pd.DataFrame(columns=['description', 'priority'])

# Load pre-existing tasks from a CSV file (if any)
try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

# Function to save tasks to a CSV file
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

# Train the task priority classifier (only if data is valid)
model = None

if not tasks.empty and tasks['description'].dropna().str.strip().any():
    vectorizer = CountVectorizer()
    clf = MultinomialNB()
    model = make_pipeline(vectorizer, clf)
    try:
        model.fit(tasks['description'], tasks['priority'])
    except ValueError as e:
        print(f"⚠️ Model training failed: {e}")
        model = None
else:
    print("⚠️ Not enough data to train the ML model.")

# Function to add a task to the list
def add_task(description, priority):
    global tasks
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()

# Function to remove a task by description
def remove_task(description):
    global tasks
    tasks = tasks[tasks['description'] != description]
    save_tasks()

# Function to list all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print("\nCurrent Tasks:")
        print(tasks)

# Function to recommend a task based on machine learning
def recommend_task():
    if not tasks.empty:
        high_priority_tasks = tasks[tasks['priority'].str.lower() == 'high']
        if not high_priority_tasks.empty:
            random_task = random.choice(high_priority_tasks['description'].tolist())
            print(f"\n✅ Recommended task: {random_task} - Priority: High")
        else:
            print("\n⚠️ No high-priority tasks available for recommendation.")
    else:
        print("\n⚠️ No tasks available for recommendations.")

# Main menu
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ").strip()
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        if description and priority in ['Low', 'Medium', 'High']:
            add_task(description, priority)
            print("✅ Task added successfully.")
        else:
            print("⚠️ Invalid input. Please enter a valid description and priority.")

    elif choice == "2":
        description = input("Enter task description to remove: ").strip()
        if description:
            remove_task(description)
            print("✅ Task removed successfully.")
        else:
            print("⚠️ Please provide a valid task description.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("👋 Goodbye!")
        break

    else:
        print("❌ Invalid option. Please select a valid number (1–5).")
