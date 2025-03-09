import yfinance as yf
import logging
from langchain.tools import Tool
import warnings
import language_tool_python

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

class StockChatBot:
    def __init__(self):
        # Initialize language tool for grammar and spelling correction
        self.language_tool = language_tool_python.LanguageTool('en-US')
        
        # Initialize stock tool
        self.stock_tool = Tool(
            name="StockPrice",
            func=self.get_stock_price,
            description="Fetches the current stock price using Yahoo Finance.",
            return_direct=True
        )
        
    def get_stock_price(self, ticker: str):
        """
        Fetch the current closing price of a given stock using Yahoo Finance.
        
        :param ticker: The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        :return: The current stock price or an error message.
        """
        ticker = ticker.strip().upper().replace("'", "").replace('"', "")
        logging.info(f"Fetching stock data for: {ticker}")
        
        stock = yf.Ticker(ticker)
        
        if stock.info.get("regularMarketPrice") is None:
            logging.error(f"Stock {ticker} is unavailable or removed from Yahoo Finance.")
            return f"Stock {ticker} is unavailable or might have been removed from Yahoo Finance."
        
        history = stock.history(period="1d")
        
        if history.empty:
            logging.warning(f"No available data for {ticker}. Market might be closed.")
            return f"No available data for {ticker}. The market might be closed or insufficient data exists."
        
        price = history["Close"].iloc[-1]
        logging.info(f"Stock data retrieved successfully for {ticker}. Price: {price:.2f} USD")
        return f"The current price of {ticker} is {price:.2f} USD."
    
    def correct_input(self, user_input: str) -> str:
        """Correct grammar and spelling in the user's input."""
        matches = self.language_tool.check(user_input)
        corrected_input = language_tool_python.utils.correct(user_input, matches)
        return corrected_input

    def start_chat(self):
        """Start a conversation with the user."""
        print('Hello! I can help you with stock price information. Feel free to ask anything.')
        
        while True:
            user_input = input('You: ').strip()
            
            # Exit condition
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print('Goodbye! Have a great day!')
                break
            
            # Correct user's input
            corrected_input = self.correct_input(user_input)
            
            if corrected_input != user_input:
                print(f'Bot: Did you mean: "{corrected_input}"? (Corrected for clarity)')
            
            # Use the tool to process the corrected input
            response = self.stock_tool.func(corrected_input)
            print(f'Bot: {response}')


if __name__ == '__main__':
    bot = StockChatBot()
    bot.start_chat()
