"""
Basic authentication for Streamlit app.

Uses environment variables for credentials.
"""

import hashlib
import os
import streamlit as st


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def check_credentials(username: str, password: str) -> bool:
    """
    Check if credentials match environment variables.

    Args:
        username: Username to check
        password: Password to check

    Returns:
        True if credentials are valid
    """
    expected_username = os.getenv('LLM_TEST_USERNAME')
    expected_password_hash = os.getenv('LLM_TEST_PASSWORD_HASH')

    # If no credentials set, authentication is disabled
    if not expected_username or not expected_password_hash:
        return True

    if username != expected_username:
        return False

    password_hash = hash_password(password)
    return password_hash == expected_password_hash


def show_login_page():
    """Display login page and return True if authenticated."""
    if 'authenticated' in st.session_state and st.session_state.authenticated:
        return True

    st.title("🔐 LLM Test Platform - Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_credentials(username, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password")
            st.stop()

    return False


def require_auth():
    """
    Require authentication to access the app.

    Call this at the beginning of main().
    """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        if not show_login_page():
            st.stop()
