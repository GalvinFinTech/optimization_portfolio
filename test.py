import streamlit as st

st.markdown("""
<style>
.dark-table {
  width: 100%;
  border-collapse: collapse;
  background-color: #000;
  color: #fff;
}
.dark-table th, .dark-table td {
  border: 1px solid #444;
  padding: 8px;
}
</style>
""", unsafe_allow_html=True)

html_table = """
<table class="dark-table">
  <thead>
    <tr><th>Mã CP</th><th>Tỷ trọng</th></tr>
  </thead>
  <tbody>
    <tr><td>FPT</td><td>60.81%</td></tr>
    <tr><td>VCI</td><td>19.59%</td></tr>
  </tbody>
</table>
"""

st.markdown(html_table, unsafe_allow_html=True)
