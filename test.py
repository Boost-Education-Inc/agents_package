from boostEdu.agents import Tutor
import os

os.environ["DB_USERNAME"] = "juansebastian"
os.environ["DB_PASSWORD"] = "VBxwUiGBv9fHN1AW"
os.environ["DB_NAME"] = "boost_test"

os.environ["BASE_URL"] = "https://questions2.openai.azure.com/"
os.environ["API_KEY"] = "63ad729069e24c99b3fe9207aa7c1bad"
os.environ["DEPLOYMENT_NAME"] = "Questions-40"

tutor = Tutor(student_id="576767676",content_id="1234")

tutor.createPresentation(" "," ")

# from test_folder import hello

# hello.printHello()