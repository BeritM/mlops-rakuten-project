import streamlit as st
import requests
import json
import time

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
# For prediction form
if 'designation_input' not in st.session_state:
    st.session_state.designation_input = ""
if 'description_input' not in st.session_state:
    st.session_state.description_input = ""
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'warning' not in st.session_state:
    st.session_state.warning = None
if 'show_prediction_message' not in st.session_state:
    st.session_state.show_prediction_message = False
if "show_upload_message" not in st.session_state:
    st.session_state.show_upload_message = False
if 'selected_category_from_dropdown_index' not in st.session_state: # To control selectbox index
    st.session_state.selected_category_from_dropdown_index = 0
if 'confirmed_category' not in st.session_state:
    st.session_state.confirmed_category = None

# --- global variables ---
SELECT_TEXT = "-- Select a category --"
#OPTIONS_FOR_DROPDOWN = [SELECT_TEXT, st.session_state.last_prediction, "--- Choose another category ---"] if st.session_state.last_prediction else [SELECT_TEXT]

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
    st.session_state.show_prediction_message = False
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
    
# --- UI Callback Functions ---

def handle_predict_button_click():
    """
    Callback for the predict button.
    Resets dropdown and handles prediction logic.
    """
    # Set dropdown to default "Select a class" by resetting its index
    st.session_state.selected_category_from_dropdown_index = 0
    st.session_state.confirmed_category = None # Clear any previous confirmation
    st.session_state.show_prediction_message = False # Reset message visibility

    designation = st.session_state.designation_input
    description = st.session_state.description_input

    if designation:
        predicted_class = get_prediction(designation, description)
        if predicted_class:
            st.session_state.last_prediction = predicted_class
            st.session_state.selected_category_from_dropdown_index = 1 # Set dropdown to predicted class
            st.session_state.show_prediction_message = True # Show prediction message
           
        else:
            st.session_state.last_prediction = None
            st.session_state.show_prediction_message = False
    else:
        st.session_state.warning = "Please enter at least the **Designation**."

def handle_confirm_category_click(selected_category_value):
    """
    Callback for the confirm category button.
    Stores confirmed category and clears input fields/dropdown.
    """
    # Get the currently selected value from the selectbox
    current_selection = selected_category_value

    if current_selection != SELECT_TEXT:
        st.session_state.show_upload_message = True
        st.session_state.confirmed_category = current_selection

        #st.write(f"Callback erreicht für Kategorie: {st.session_state.confirmed_category}")

        #with st.status(label="Uploading article...", state="running", expanded=False) as status:
        #    time.sleep(1)
        #    status.update(label=f"Article uploaded in category **'{st.session_state.confirmed_category}'**", state='complete', expanded=False)
        #    time.sleep(2)

        # a) Clear input fields and reset dropdown after confirmation
        st.session_state.designation_input = ""
        st.session_state.description_input = ""
        st.session_state.last_prediction = None # Clear prediction
        st.session_state.selected_category_from_dropdown_index = 0 # Reset dropdown
        st.session_state.show_prediction_message = False # Hide prediction message


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

        st.subheader("Model Information")
        if st.button("Fetch Model Info"):
            model_info = get_model_info()
            if model_info:
                st.json(model_info) # Display model info in a nice JSON format


    # --- Prediction Section (for all logged-in users) ---
    st.header("Product Type Prediction")
    st.markdown("---")

    st.subheader("Get Prediction")
    with st.form("prediction_form"):
        designation = st.text_area("Designation (required)", 
                                   value=st.session_state.designation_input,
                                   key="designation_input",
                                   help="e.g., 'Voiture miniature 1976 Ford Mustang'")
        description = st.text_area("Description (optional)", 
                                   value=st.session_state.description_input,
                                   key="description_input",
                                   help="e.g., 'couleur bleu.'")
        predict_button = st.form_submit_button("Confirm", on_click=handle_predict_button_click)

    if 'warning' in st.session_state and st.session_state.warning:
        st.warning(st.session_state.warning)
        st.session_state.warning = None
    
    if st.session_state.show_prediction_message:
        st.info("Please select product category from the dropdown below.")

    # --- Category Selection Section --- (after prediction)
    if st.session_state.show_prediction_message and st.session_state.last_prediction:
        st.markdown("---")

        OPTIONS_FOR_DROPDOWN = [SELECT_TEXT, st.session_state.last_prediction, "--- Choose another category ---"]

        current_selection = st.selectbox(
            "Category selection",
            options=OPTIONS_FOR_DROPDOWN,
            index=st.session_state.selected_category_from_dropdown_index,
            key="category_select"
        )

        if current_selection != SELECT_TEXT:
            st.button("Upload article", 
                    key="confirm_category_button", 
                    on_click=handle_confirm_category_click,
                    args=(current_selection,))
        
        #message_placeholder = st.empty()

        #if st.session_state.get('show_upload_message'):
        #    if time.time() < st.session_state.upload_message_end_time:
        #        if st.session_state.get('confirmed_category'):
        #            message_placeholder.success(f"Article uploaded in category **'{st.session_state.confirmed_category}'**")

        #    else:
        #        message_placeholder.empty()
        #        st.session_state.show_upload_message = False
        #        st.session_state.upload_message_end_time = 0
        #        st.rerun()
    message_placeholder = st.empty()
    if 'show_upload_message' in st.session_state and st.session_state.show_upload_message:    
        if st.session_state.get('confirmed_category'):
            message_placeholder.success(f"Article uploaded in category **'{st.session_state.confirmed_category}'**")
            time.sleep(3)
            message_placeholder.empty()
            st.session_state.show_upload_message = False


    

st.markdown("---")
st.caption("Developed with Streamlit and FastAPI")
