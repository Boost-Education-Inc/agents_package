import json
import os
import uuid
import logging
logging.getLogger().setLevel(logging.INFO)
import asyncio
from datetime import datetime
from typing import AsyncIterable

from langchain.callbacks import AsyncIteratorCallbackHandler
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain.vectorstores import Vectara
from langchain.chat_models import ChatOpenAI

from boostEdu.prompt_templates import *
from boostEdu.prompt_templates_student import *
from boostEdu.prompt_templates_content_expert import *


class Agent():
    def __init__(self,agid,is_streaming=False):
        self.is_streaming=is_streaming
        self.agid=agid
        self.MemoryDB= self._initMemoryDB()
        self.llm=self._initLLM()
    
    def _initMemoryDB(self):
        MONGO_URL = f"mongodb+srv://{os.environ.get('DB_USERNAME')}:{os.environ.get('DB_PASSWORD')}@cluster0.yvvgepo.mongodb.net/?retryWrites=true&w=majority"
        MONGO_CLIENT = MongoClient(MONGO_URL, server_api=ServerApi('1'))
        return MONGO_CLIENT[os.environ.get('DB_NAME')]

    def _initLLM(self):
        OPENAI_KEY = os.environ.get("OPEN_AI_KEY")
        MODEL_NAME = os.environ.get("OPEN_AI_MODEL_NAME")
        if self.is_streaming:
            callback = AsyncIteratorCallbackHandler()
            return ChatOpenAI(
            openai_api_key=OPENAI_KEY,
            model=MODEL_NAME,
            temperature= 0.1,
            streaming=True,
            callbacks=[callback])
        else:
            return ChatOpenAI(
            openai_api_key=OPENAI_KEY,
            model=MODEL_NAME,
            temperature= 0.1)
  
  
class AWSManager():
    def __init__(self,apigw_client=None,connection_id=None,s3=None,polly_client=None):
        logging.debug(f"ℹ️🔔Init AWS Manager")
        self.apigw_client=apigw_client
        self.connection_id=connection_id
        
        self.s3=s3
        self.polly_client=polly_client
    
    def sendDataToClientAPIGateway(self,data):
        response = self.apigw_client.post_to_connection(
            ConnectionId=self.connection_id,
            Data=json.dumps(data, ensure_ascii=False)
        )
        logging.debug(f"ℹ️🔔Response: {response}")    
    
      
    def getSpeechBytes(self,script):
        response =  self.polly_client.synthesize_speech(VoiceId='Gregory',
                                                   OutputFormat='mp3',
                                                   Engine="long-form",
                                                   Text = script)
        file_bytes= response['AudioStream'].read()
        return file_bytes
        
    def savePollyIntoS3(self,file_bytes):
        bucket_name = 'boostfs'
        folder_name = 'temp'
        file_path = f'{folder_name}/{str(uuid.uuid4())}.mp3'
        self.s3.put_object(Body=file_bytes, Bucket=bucket_name, Key=file_path)
        return f'https://{bucket_name}.s3.amazonaws.com/{file_path}' 
            
             
class Student(Agent):               
    def __init__(self,student_id,is_streaming=False):
        super().__init__(student_id,is_streaming)
        self.workingMemory= self._initWorkingMemory()
        self.longMemory=[]
        
    def _initWorkingMemory(self):
        collection = self.MemoryDB["students"]
        projection = {"name": 1, "age": 1,"gender":1, "description": 1,"_id":0} 
        memory_document = collection.find_one({"_id":self.agid},projection)
        if memory_document is None:
           raise Exception("Student not found")
        return f"{memory_document}"
          
    def initLongMemory(self,content_id):
        collection = self.MemoryDB["long_term_memories"]
        memory_document = collection.find_one({"student_id":self.agid,"content_id":content_id})
        if memory_document is None:
          collection.insert_one({"_id":str(uuid.uuid4()),"student_id":self.agid,"content_id":content_id,"memory":[]})
          memory_document = collection.find_one({"student_id":self.agid,"content_id":content_id})
        self.longMemory=memory_document["memory"]

    def retrieve(self,perception):
        formatted_prompt = STUDENT_RETRIEVE_TEMPLATE.format(student_data=self.workingMemory,long_memory="\n".join(self.longMemory),question=perception)
        logging.debug(formatted_prompt)
        return self.llm.predict(formatted_prompt)
        
 
class ContentExpert(Agent):
    def __init__(self,agid=None,ret_params=None,is_streaming=False):
        if (agid==None and ret_params==None): raise Exception("You need to provide either agid or ret_params")
        super().__init__(agid,is_streaming)
        self.workingMemory= self._initWorkingMemory(ret_params)
        
    def _initWorkingMemory(self,ret_params):
        params= self._getRetrieverParams(ret_params)
        logging.debug(f"🔔Retriever params: {params}")
        if params["type"]=='vectara':
            logging.debug(f"🟢 Is Vectara")
            vectara = Vectara(
                vectara_customer_id=params["keys"]["vectara_customer_id"],
                vectara_corpus_id=params["keys"]["vectara_corpus_id"],
                vectara_api_key=params["keys"]["vectara_api_key"],
            )
            retriever = vectara.as_retriever(search_type=params["search_type"],search_kwargs=params["search_kwargs"])
        else:
            logging.debug(f"🔴Not Vectara")
        return retriever    
        
    def _getRetrieverParams(self,ret_params):
        collection = self.MemoryDB["content_agents"]
        if self.agid==None:
            self.agid=str(uuid.uuid4())
            collection.insert_one({"_id":self.agid,"params":ret_params})
        query = {"_id": self.agid}
        projection = {"params": 1,"_id":0}
        logging.debug(f"🔔Agent id: {self.agid}")
        result = collection.find_one(query, projection)
        return result["params"]
    
    def retrieve(self,perception):
        content_chunks = "".join(document.page_content for document in self.workingMemory.invoke(input="Key aspects of each part/charper/section"))
        formatted_prompt = CONTENT_AGENT_ANSW_QUESTION_CONTENT_TEMPLATE.format(content_chunks=content_chunks,question=perception)
        logging.debug(formatted_prompt)
        return self.llm.predict(formatted_prompt)

 
class Tutor(Agent):
    def __init__(self,is_streaming=False):
        super().__init__(None,is_streaming)
               
    def retrieve(self,studentData,studentLongMemory,content,perception,awsManager=None):
        formatted_prompt = TUTOR_CONTEXT_TEMPLATE.format(student_data=studentData,
                                                         student_background=studentLongMemory,
                                                         content_data=content,
                                                         question=perception)
        logging.debug(formatted_prompt)
        if (self.is_streaming==False):
            output= self.llm.predict(formatted_prompt)
        else:
            output=  asyncio.run(self._sendStreamingResponse(awsManager,formatted_prompt))
        logging.debug(output)
        if (self.is_streaming==False): return output 
        
   
    def retrievePresentation(self,studentData,studentLongMemory,content,awsManager:AWSManager=None):
        formatted_prompt = TUTOR_PRESENTATION_TEMPLATE.format(student_data=studentData,
                                                         student_background=studentLongMemory,
                                                         content_data=content)
        logging.debug(formatted_prompt)
        presentation_code=self.llm.predict(formatted_prompt)
        presentation_code=presentation_code.replace("html","")
        presentation_code=presentation_code.replace("\n","")
        if (awsManager==None):return presentation_code
        else:awsManager.sendDataToClientAPIGateway(presentation_code)
    
    
    def retrieveLearningPlan(self,studentData,studentLongMemory,content,awsManager:AWSManager=None):
        formatted_prompt = TUTOR_PLAN_TEMPLATE.format(student_data=studentData,
                                                         student_background=studentLongMemory,
                                                         content_data=content)
        logging.debug(formatted_prompt)
        plan_code=self.llm.predict(formatted_prompt)
        plan_code=plan_code.replace("html","")
        if (awsManager==None):return plan_code
        else:awsManager.sendDataToClientAPIGateway(plan_code)
    
    
    def retrievePresentationToSpeech(self,presentation_html_str,awsManager:AWSManager,sendToClientAPIGateway=False):
        formatted_prompt = TUTOR_PRESENTATION_SCRIPT_TEMPLATE.format(presentation_html=presentation_html_str)
        presentation_script=self.llm.predict(formatted_prompt)
        logging.debug(presentation_script)
        data={
            "audio_url":awsManager.savePollyIntoS3(awsManager.getSpeechBytes(presentation_script)),
            "presentation_script":presentation_script}
        
        if sendToClientAPIGateway: awsManager.sendDataToClientAPIGateway(data)
        else: return data

    
    async def _sendStreamingResponse(self,awsManager:AWSManager,input_prompt):
        fullOutput=""
        async for value in self._getTutorAsyncResponse(input_prompt):
            if (awsManager==None):
                logging.debug(value)
            else:
                awsManager.sendDataToClientAPIGateway(value)
            if(value!="[¬TUTOR_END¬]"):fullOutput=fullOutput+value
        return  fullOutput
        
           
    async def _getTutorAsyncResponse(self,prompt) -> AsyncIterable[str]:
        task = asyncio.create_task(self.llm.apredict(prompt))
        try:
            async for token in self.llm.callbacks[0].aiter():
                yield token
        except Exception as e:
            print(f"Caught exception: {e}")
        finally:
            yield "[¬TUTOR_END¬]"
            self.llm.callbacks[0].done.set()
        await task
