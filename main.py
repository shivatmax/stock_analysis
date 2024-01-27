import streamlit as st
from crewai import Crew
from textwrap import dedent
from stock_analysis_agents import StockAnalysisAgents
from stock_analysis_tasks import StockAnalysisTasks
from dotenv import load_dotenv
load_dotenv()

class FinancialCrew:
  def __init__(self, company):
    self.company = company

  def run(self):
    agents = StockAnalysisAgents()
    tasks = StockAnalysisTasks()

    research_analyst_agent = agents.research_analyst()
    financial_analyst_agent = agents.financial_analyst()
    investment_advisor_agent = agents.investment_advisor()

    research_task = tasks.research(research_analyst_agent, self.company)
    financial_task = tasks.financial_analysis(financial_analyst_agent)
    filings_task = tasks.filings_analysis(financial_analyst_agent)
    recommend_task = tasks.recommend(investment_advisor_agent)

    crew = Crew(
      agents=[
        research_analyst_agent,
        financial_analyst_agent,
        investment_advisor_agent
      ],
      tasks=[
        research_task,
        financial_task,
        filings_task,
        recommend_task
      ],
      verbose=True
    )

    result = crew.kickoff()
    return result

def main():
  st.title("Financial Analysis Crew")
  st.write('-------------------------------')
  company = st.text_input("What is the company you want to analyze?")

  if st.button("Run Analysis"):
    financial_crew = FinancialCrew(company)
    result = financial_crew.run()
    st.write("\n\n########################")
    st.write("## Here is the Report")
    st.write("########################\n")
    st.write(result)
    st.balloons()

if __name__ == "__main__":
  main()
