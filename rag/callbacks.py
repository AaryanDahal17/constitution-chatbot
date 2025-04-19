from langchain.callbacks.base import BaseCallbackHandler
from ui.styling import get_professional_style

class StreamHandler(BaseCallbackHandler):
    """Custom callback handler for streaming LLM responses to Streamlit"""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.style = get_professional_style()
        self.close_div = '</div>'

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Process each new token from the LLM"""
        self.text += token
        self.container.markdown(
            f"{self.style}{self.text}▌{self.close_div}",
            unsafe_allow_html=True
        ) 