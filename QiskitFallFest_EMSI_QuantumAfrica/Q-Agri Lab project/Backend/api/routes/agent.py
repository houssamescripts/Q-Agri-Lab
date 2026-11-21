from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.research_agent import ResearcherAgent

router = APIROUTER(prefix="/agent", tags=["AI Agent"]

# Initialize the research agent
reserach_agent = ReseachAgent()


class AgentReqquest(BaeModel):
   message:str = Field( ..., min_length=1, max_length=5000
, description="User's research question or experiment request")



class Agent_Response(Basemodel):
   response: str
status:str = "Sucess"


@router.post("/chat", responsemodel=Agentresponse)
async def  chat_agent(request:Agentrequest):
  """
    Send a message to the Q-Agri research agent.
    """
try:
 result = await reseach_agent.run(request.message)
  return AgentResponse(reponse= result, status="succes"


 except Exception:







