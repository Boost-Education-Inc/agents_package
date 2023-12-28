import json
import os
import uuid
import logging
import asyncio
from datetime import datetime
from typing import AsyncIterable

from langchain.callbacks import AsyncIteratorCallbackHandler
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain.vectorstores import Vectara
from langchain.chat_models import ChatOpenAI

from boostEdu.prompt_templates import TUTOR_PRESENTATION_TEMPLATE,TUTOR_PLAN_TEMPLATE,TUTOR_CONTEXT_TEMPLATE,TUTOR_PRESENTATION_SCRIPT_TEMPLATE

logging.getLogger().setLevel(logging.DEBUG)

class Agent():
    def __init__(self,is_streaming=False):
        self.DB= self._initDB()
        self.is_streaming=is_streaming
        self.llm=self._initLLM()
    
    def _initDB(self):
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


class Tutor(Agent):
    def __init__(self,student_id,content_id,is_streaming=False):
        super().__init__(is_streaming)
        ###IDS####
        self.student_id=student_id
        self.content_id=content_id
        
        ########################
        
        self.contentRetriever = self._initContentRetriever()
        self.allInteractionsMemory, self.longTermMemory =self._initMemory()
   
    
    def _initContentRetriever(self):
            os.environ["VECTARA_CUSTOMER_ID"] = "3566257016"
            os.environ["VECTARA_CORPUS_ID"] = "3"
            os.environ["VECTARA_API_KEY"] = "zwt_1JDDeIhw6YcfPnaW7VZotYx_lkbRqCMtTLzdcQ"
            vectara = Vectara()
            retriever = vectara.as_retriever(lambda_val=0.025, k=5, filter=None)
            return retriever 
    
    
    
    def _initMemory(self):
        collection = self.DB["interactions_memories"]
        memory_document = collection.find_one({"student_id":self.student_id,"content_id":self.content_id})
        if memory_document is None:
          collection.insert_one({"_id":str(uuid.uuid4()),"student_id":self.student_id,"content_id":self.content_id,"all_interactions_memory":[],"long_term_memory":[]})
          memory_document = collection.find_one({"student_id":self.student_id,"content_id":self.content_id})
        return memory_document["all_interactions_memory"],memory_document["long_term_memory"]


    def ask(self,prompt,apigw_client=None,connection_id=None):
        studentData= self._getStudentData()
        content_chunk = "".join(document.page_content for document in self.contentRetriever.invoke(input=prompt))
        formatted_prompt = TUTOR_CONTEXT_TEMPLATE.format(chat_history="\n".join(self.allInteractionsMemory),context=content_chunk,question=prompt,student_data=studentData)
        if (self.is_streaming==False):
            output= self.llm.predict(formatted_prompt)
        else:
            output=  asyncio.run(self._sendStreamingResponse(apigw_client,connection_id,formatted_prompt))
        logging.warning(output)
        self._updateAllInteractionsMemory(str({"type":"human","content":prompt,"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}))
        self._updateAllInteractionsMemory(str({"type":"ai","content":output,"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}))
        
        if (self.is_streaming==False): return output
    
    
    def createPresentation(self,apigw_client=None,connection_id=None):
        studentData= self._getStudentData()
        #content_query="Key aspects of each part/charper/section"
        #content_query="Intro of each part/charper/section"
        content_query="What is the main idea"
        content_chunk = "".join(document.page_content for document in self.contentRetriever.invoke(input=content_query))
        formatted_prompt = TUTOR_PRESENTATION_TEMPLATE.format(chat_history="\n".join(self.allInteractionsMemory),context=content_chunk,student_data=studentData)
        logging.warning(formatted_prompt)
        presentation_code=self.llm.predict(formatted_prompt)
        presentation_code=presentation_code.replace("html","")
        presentation_code=presentation_code.replace("\n","")
        if (apigw_client==None or connection_id==None):
            return presentation_code
        else:
            self._sendDataToClient(apigw_client,connection_id,presentation_code)
     
     
    def createLearningPlan(self,apigw_client=None,connection_id=None):
        studentData= self._getStudentData()
        content_query="Key aspects of each part/charper/section"
        content_chunk = "".join(document.page_content for document in self.contentRetriever.invoke(input=content_query))
        formatted_prompt = TUTOR_PLAN_TEMPLATE.format(chat_history="\n".join(self.allInteractionsMemory),context=content_chunk,student_data=studentData)
        logging.warning(formatted_prompt)
        plan_code=self.llm.predict(formatted_prompt)
        plan_code=plan_code.replace("html","")
        if (apigw_client==None or connection_id==None):
            return plan_code
        else:
            self._sendDataToClient(apigw_client,connection_id,plan_code)  
     
    def createPresentationScript(self,presentation_html_str,apigw_client,connection_id,s3,polly_client):
        formatted_prompt = TUTOR_PRESENTATION_SCRIPT_TEMPLATE.format(presentation_html=presentation_html_str)
        presentation_script=self.llm.predict(formatted_prompt)
        logging.warning(presentation_script)
        file_bytes=self.speak(presentation_script,polly_client)
        audio_url=self._savePollyIntoS3(file_bytes,s3)
        self._sendDataToClient(apigw_client,connection_id,{"audio_url":audio_url,"presentation_script":presentation_script})
    
    
    def speak(self,script,polly_client):
        response =  polly_client.synthesize_speech(VoiceId='Gregory',
                                                   OutputFormat='mp3',
                                                   Engine="long-form",
                                                   Text = script)
        file_bytes= response['AudioStream'].read()
        return file_bytes
       
        
    def _savePollyIntoS3(self,file_bytes,s3):
        bucket_name = 'boostfs'
        folder_name = 'temp'
        file_path = f'{folder_name}/{str(uuid.uuid4())}.mp3'
        s3.put_object(Body=file_bytes, Bucket=bucket_name, Key=file_path)
        return f'https://{bucket_name}.s3.amazonaws.com/{file_path}'
     
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

    async def _sendStreamingResponse(self,apigw_client,connection_id,input_prompt):
        fullOutput=""
        async for value in self._getTutorAsyncResponse(input_prompt):
            if (apigw_client==None or connection_id==None):
                logging.warning(value)
            else:
                self._sendDataToClient(apigw_client,connection_id,value)
            if(value!="[¬TUTOR_END¬]"):fullOutput=fullOutput+value
        return  fullOutput      
     
        
    def _getStudentData(self):
        collection = self.DB["students"]
        query = {"_id": self.student_id}
        projection = {"name": 1, "age": 1, "description": 1,"_id":0} 
        result = collection.find_one(query, projection)
        return str(result)
    
    
    def _updateAllInteractionsMemory(self,newInteraction):
        collection = self.DB["interactions_memories"]
        self.allInteractionsMemory.insert(0,newInteraction)
        filter_criteria = {"student_id": self.student_id, "content_id": self.content_id}
        update_operation = {"$set": {"all_interactions_memory":self.allInteractionsMemory}}
        collection.update_one(filter_criteria, update_operation)
    
            
    def _sendDataToClient(self,apigw_client,connection_id, data):
        response = apigw_client.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(data, ensure_ascii=False)
        )
        logging.warning(f"ℹ️🔔Response: {response}")



class ContentAgent(Agent):
    def __init__(self,agid=None,ret_params=None,is_streaming=False):
        super().__init__(is_streaming)
        if (agid==None and ret_params==None): raise Exception("You need to provide either agid or ret_params")
        self.agid=agid
        #self.retriever_params=ret_params
        self._getRetrieverFromParams(ret_params)
      
    def _getRetrieverFromParams(self,ret_params):
        params= self._getRetrieverParams(ret_params)
        logging.info(f"🔔Retriever params: {params}")
        if params["type"]=='vectara':
            logging.info(f"🟢 Is Vectara")
        else:
            logging.info(f"🔴Not Vectara")    
        
    def _getRetrieverParams(self,ret_params):
        collection = self.DB["content_agents"]
        if self.agid==None:
            self.agid=str(uuid.uuid4())
            collection.insert_one({"_id":self.agid,"params":ret_params})
        query = {"_id": self.agid}
        projection = {"params": 1,"_id":0}
        logging.info(f"🔔Agent id: {self.agid}")
        result = collection.find_one(query, projection)
        return result["params"]


#projection = {"vectara_customer_id": 1, "vectara_corpus_id": 1, "vectara_api_key": 1,"_id":0}'       

# class ContentAgent(Agent):
#     def __init__(self,vectara_customer_id, vectara_corpus_id, vectara_api_key,k=5,search_type='similarity',is_streaming=False):
#         super().__init__(is_streaming)
#         self.contentRetriever = self._initContentRetriever(vectara_customer_id, vectara_corpus_id, vectara_api_key,k,search_type)
    
#     def _initContentRetriever(self, vectara_customer_id, vectara_corpus_id, vectara_api_key,k,search_type):
#         vectara = Vectara(
#             vectara_customer_id= vectara_customer_id,
#             vectara_corpus_id= vectara_corpus_id,
#             vectara_api_key= vectara_api_key
#         )
#         retriever = vectara.as_retriever(search_type=search_type,search_kwargs={"k":k})
#         return retriever 


{
    "_id":"fad",
    "params":{
        "type":"vectara",
        "vectara_customer_id": "3566257016",
    }
}    