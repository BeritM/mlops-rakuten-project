import streamlit as st
import requests
import json

# --- Configuration ---
# IMPORTANT: Ensure these URLs match where your FastAPI services are running.
AUTH_API_BASE_URL = "http://localhost:8001"
PREDICT_API_BASE_URL = "http://localhost:8002" # Updated to 8002 as per error message

# --- Session State Initialization ---
# Initialize session state variables if they don't already exist.
# This helps maintain state across Streamlit re-runs.
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

# --- Helper Functions for API Calls ---

def login_user(username, password):
    """
    Attempts to log in a user by sending credentials to the auth API.
    Updates session state upon successful login.
    """
    try:
        response = requests.post(
            f"{AUTH_API_BASE_URL}/login",
            data={"username": username, "password": password}
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        st.session_state.logged_in = True
        st.session_state.access_token = data["access_token"]
        st.session_state.username = username
        # For this example, we'll assume the role is 'admin' if username is 'admin', otherwise 'user'.
        # In a production app, you would parse the JWT payload to extract the role.
        if username == "admin":
            st.session_state.user_role = "admin"
        else:
            st.session_state.user_role = "user"

        st.success(f"Logged in as {username}!")
        st.rerun() # Rerun to update UI based on login status
    except requests.exceptions.RequestException as e:
        st.error(f"Login failed: {e}")
        if response.status_code == 401:
            st.error("Incorrect username or password.")
        else:
            st.error(f"An error occurred: {response.text}")

def logout_user():
    """Logs out the user by clearing session state."""
    st.session_state.logged_in = False
    st.session_state.access_token = None
    st.session_state.username = None
    st.session_state.user_role = None
    st.info("Logged out successfully.")
    st.rerun() # Rerun to update UI based on logout status

def create_user(username, password, role):
    """
    Sends a request to the auth API to create a new user.
    Requires admin privileges. Uses Authorization: Bearer.
    """
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        response = requests.post(
            f"{AUTH_API_BASE_URL}/users",
            json={"username": username, "password": password, "role": role},
            headers=headers
        )
        response.raise_for_status()
        st.success(f"User '{username}' created successfully.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create user: {e}")
        st.error(f"Error details: {response.text}")

def delete_user(username):
    """
    Sends a request to the auth API to delete a user.
    Requires admin privileges. Uses Authorization: Bearer.
    """
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        response = requests.delete(
            f"{AUTH_API_BASE_URL}/users/{username}",
            headers=headers
        )
        response.raise_for_status()
        st.success(f"User '{username}' deleted successfully.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to delete user: {e}")
        st.error(f"Error details: {response.text}")

def get_prediction(designation, description):
    """
    Sends a prediction request to the predict API.
    Requires authentication. Uses 'token' header.
    """
    # Prediction API expects 'token' header, not 'Authorization: Bearer'
    headers = {"token": st.session_state.access_token}
    try:
        response = requests.post(
            f"{PREDICT_API_BASE_URL}/predict",
            json={"designation": designation, "description": description},
            headers=headers
        )
        response.raise_for_status()
        return response.json()["predicted_class"]
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {e}")
        st.error(f"Error details: {response.text}")
        return None

def get_model_info():
    """
    Retrieves model information from the predict API.
    Requires authentication. Uses 'token' header.
    """
    # Prediction API expects 'token' header, not 'Authorization: Bearer'
    headers = {"token": st.session_state.access_token}
    try:
        response = requests.get(
            f"{PREDICT_API_BASE_URL}/model-info",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to retrieve model info: {e}")
        st.error(f"Error details: {response.text}")
        return None

# --- Streamlit UI Layout ---

st.set_page_config(page_title="API Interaction Dashboard", layout="centered")

st.title("API Interaction Dashboard")

# --- Authentication Section ---
st.header("Authentication")

if not st.session_state.logged_in:
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            login_user(username, password)
else:
    st.success(f"You are logged in as **{st.session_state.username}** (Role: **{st.session_state.user_role}**)")
    st.button("Logout", on_click=logout_user)

    # --- Admin Section (only for admins) ---
    if st.session_state.user_role == "admin":
        st.header("Admin Panel (User Management)")
        st.markdown("---")

        st.subheader("Create New User")
        with st.form("create_user_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"], index=0)
            create_user_button = st.form_submit_button("Create User")

            if create_user_button:
                create_user(new_username, new_password, new_role)

        st.subheader("Delete User")
        with st.form("delete_user_form"):
            user_to_delete = st.text_input("Username to Delete")
            delete_user_button = st.form_submit_button("Delete User")

            if delete_user_button:
                delete_user(user_to_delete)

    # --- Prediction Section (for all logged-in users) ---
    st.header("Product Type Prediction")
    st.markdown("---")

    st.subheader("Get Prediction")
    with st.form("prediction_form"):
        designation = st.text_area("Designation", help="e.g., 'Smartphone XYZ'")
        description = st.text_area("Description", help="e.g., 'Latest model with 128GB storage and 5G connectivity.'")
        predict_button = st.form_submit_button("Get Prediction")

        if predict_button:
            if designation and description:
                predicted_class = get_prediction(designation, description)
                if predicted_class:
                    st.success(f"Predicted Product Class: **{predicted_class}**")
            else:
                st.warning("Please enter both designation and description.")

    st.subheader("Model Information")
    if st.button("Fetch Model Info"):
        model_info = get_model_info()
        if model_info:
            st.json(model_info) # Display model info in a nice JSON format

st.markdown("---")
st.caption("Developed with Streamlit and FastAPI")
