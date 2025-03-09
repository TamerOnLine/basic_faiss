import os
import logging
import requests
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_ollama import OllamaLLM
from langchain.tools import Tool
from langchain.prompts import PromptTemplate  # تم استيراده هنا
import language_tool_python

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")


class WeatherChatBot:
    def __init__(self):
        # Initialize the language tool for grammar and spelling correction
        self.language_tool = language_tool_python.LanguageTool('en-US')
        
        # Initialize Weather Tool
        self.weather_tool = Tool(
            name="WeatherTool",
            func=self.get_weather,
            description="Fetches weather information for a given city.",
            return_direct=True,
        )
        
        # Initialize language model (Ollama)
        self.llm = OllamaLLM(model="llama3.2")
        
        # Initialize agent with weather tool
        self.agent = initialize_agent(
            [self.weather_tool],
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            llm=self.llm,
            verbose=True
        )

        # Initialize the prompt template for the weather question
        self.prompt_template = PromptTemplate(
            input_variables=["city"],
            template="What’s the weather like in {city}?"
        )

    def get_weather(self, city: str) -> str:
        """Fetch weather information for a given city."""
        if not API_KEY:
            logging.error('API_KEY is missing. Please check the .env file.')
            return 'API_KEY is missing. Please check the .env file.'

        # Use the prompt template to format the weather question
        filled_prompt = self.prompt_template.format(city=city)  # Filling in the city
        print(filled_prompt)  # Printing the formatted prompt (Optional)

        url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
        logging.info(f'Fetching weather data for city: {city}')
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an exception for HTTP errors
            data = response.json()
            
            if 'weather' in data and 'main' in data:
                weather_desc = data['weather'][0]['description']
                temp = data['main']['temp']
                logging.info(f'Weather data retrieved successfully for {city}.')
                return f'Weather in {city}: {weather_desc}, Temperature: {temp}°C.'
            
            logging.warning(f'Unexpected response format: {data}')
            return 'An error occurred while fetching data.'
        except requests.exceptions.RequestException as e:
            logging.error(f'Error fetching weather data: {e}')
            return 'Failed to retrieve weather data.'

    def correct_input(self, user_input: str) -> str:
        """Correct grammar and spelling in the user's input."""
        matches = self.language_tool.check(user_input)
        corrected_input = language_tool_python.utils.correct(user_input, matches)
        return corrected_input

    def start_chat(self):
        """Start a conversation with the user."""
        print('Hello! I can help you with weather information. Feel free to ask anything.')
        
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
            
            # Use the agent to process the corrected input
            response = self.agent.invoke(corrected_input)
            print(f'Bot: {response}')


if __name__ == '__main__':
    bot = WeatherChatBot()
    bot.start_chat()
