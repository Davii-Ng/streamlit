import streamlit as st


if 'button' not in st.session_state:
    st.session_state.button = False

def click_button():
    st.session_state.button = not st.session_state.button

st.button('click me!', on_click = click_button)

if st.session_state.button:
    st.write('Button is on')
else:
    st.write('Button off')

st.slider('Select a value:', disabled = st.session_state.button)





