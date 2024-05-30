from flask import Flask, request, jsonify, render_template
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain.agents import create_openai_tools_agent
from langchain import hub
from loaders import loadfrompdf, loadfromWeb
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')


class Chatgroq:

    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name

    def create_tool(self, pdfretriver, webretriver):
        pdf_retriver_tool = create_retriever_tool(pdfretriver, 'PDF Rag', 'Search for any question on partnership banking service use this tool')
        web_retriver_tool = create_retriever_tool(webretriver, 'Web Search', 'Search for any question apart from the pdf content use this tool to search the Web')
        tools = [pdf_retriver_tool, web_retriver_tool]
        return tools

    def create_agent(self, tools, prompt):
        llm = ChatGroq(groq_api_key=self.api_key, model_name=self.model_name)
        agent = create_openai_tools_agent(llm, tools=tools, prompt=prompt)
        return agent

    def agent_executor(self, tools, prompts):
        agent = self.create_agent(tools, prompts)
        agentExecutor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        return agentExecutor

    def get_prompt(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", '''You are a very powerful assistant, who retrieves answers to queries given from a pdf or from web. Make sure to display the response in a brief precise manner, do not just summarize. Make sure to retrieve numerical data as is. If the response is in tabular form, display it in a table. If all the content is displayed, you will get a penalty.'''),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        return prompt

    def generate_response(self):
        webR = loadfromWeb()
        pdfR = loadfrompdf('pdf/partnership_banking_services_rfp.pdf')
        prompt = self.get_prompt()
        tools = self.create_tool(pdfR, webR)
        agentexecutor = self.agent_executor(tools, prompt)
        return agentexecutor

chatbot = Chatgroq('', model_name='mixtral-8x7b-32768')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    message = data.get('message', '')
    agent = chatbot.generate_response()
    response = agent.invoke({'input': message})
    return jsonify({'output': response['output']})

if __name__ == '__main__':
    app.run(debug=True)
