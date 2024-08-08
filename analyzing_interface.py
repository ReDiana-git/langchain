from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os


def analyze_controller(qa):
    project = "ConsultationController"
    template = "You are a software engineer.Print out the content of {project} methods.Don't tell me anything other than the code."
    question = template.format(project=project)
    chat_history = []

    find_controller_result = qa({"question": question, "chat_history": chat_history})

    chat_history.append((question, find_controller_result['answer']))
    if os.environ['LANGCHAIN_DEBUG'] == '1':
        print(f"\*\*Question\*\*:\n {question} ")

        print(f"\*\*Answer\*\*:\n {find_controller_result['answer']} ")
    return find_controller_result['answer'], chat_history


def analyze_properties(qa, project_code, chat_history):
    example_code = '''
    <example code>
    @GetMapping("/appointment/consultation/{recordId}")
    public ResponseEntity<?> checkConsultation(@PathVariable("recordId") String recordId){
        CheckConsultationDTO checkConsultationDTO = consultationService.checkConsultation(recordId);
        return ResponseEntity.status(HttpStatus.OK).body(checkConsultationDTO);
    </example code>
    
    <exapmle output>
    {
        "name": "checkConsultation",
        "url": "/appointment/consultation/{recordId}",
        "input body": "null",
        "method": "get"
    }
    </example ouput>
    '''
    template = '''
    {example_code}
    1.You are a software engineer.
    2.This is a Java project based on Spring Boot.
    3.Help me find all the controller API entry points.
    4.Base on the example code to print the out example output.
    5.Only print out the name, url, input body and method.  
    6.Help me organize and print out all the method information for target_code.
    7.Don't tell me anything other than the entry points.
    
    <target_code>
    {project_code}
    </target_code>
    '''

    question = template.format(project_code=project_code, example_code=example_code)

    columnar_code = qa({"question": question, "chat_history": chat_history})

    chat_history.append((question, columnar_code['answer']))
    if os.environ['LANGCHAIN_DEBUG'] == '1':
        print(f"\*\*Question\*\*:\n {question} ")

        print(f"\*\*Answer\*\*:\n {columnar_code['answer']} ")

    return columnar_code['answer'], chat_history

def analyze_relationship(qa, result, chat_history):
    question = '''
    <example code>
    public class updateDBDTO{
        String id;
        String database;
        String consultation;
    }
    </example code>
    
    <example microservice relationship>
    {
        "testCaseName": "TestGetAppointmentByOwnerName",
        "relationship": [
            {
                "controllerName": "AppointmentControllerTest",
                "methodName": "getAppointmentByOwnerName",
                "url": "/appointment/getAppointments",
                "inputBody": "OwnerNameDTO",
                "methodType": "post",
                "dependency": [
                    {
                        "controllerName": "AppointmentController",
                        "methodName": "getRecordById",
                        "url": "/appointment/medicalRecord/{id}",
                        "inputBody": null,
                        "methodType": "get",
                        "dependency": null
                    },
                    {
                        "controllerName": "UpdateRepository",
                        "methodName": "updateConsultationDB",
                        "url": "/appointment/updateConsultationDB",
                        "inputBody": updateDBDTO,
                        "methodType": "post",
                        "dependency": null                    
                    }
                ]
            }
        ]
    }
    </example microservice relationship>
    <microservice relationship>
    {
        "testCaseName": "TestCheckConsultation",
        "relationship": {
            "controllerName": "ConsultationController",
            "methodName": "testCheckConsultation",
            "url": "/appointment/consultation/12345",
            "inputBody": null,
            "methodType": "get",
            "dependency": [
                {
                    "controllerName": "AppointmentController",
                    "methodName": "getRecordById",
                    "url": "/appointment/medicalRecord/{id}",
                    "inputBody": null,
                    "methodType": "get",
                    "dependency": null
                },
                {
                    "controllerName": "MedicineController",
                    "methodName": "getMedicine",
                    "url": "/appointment/medicine/{recordId}",
                    "inputBody": null,
                    "methodType": "get",
                    "dependency": [
                        {
                            "controllerName": "AppointmentController",
                            "methodName": "getRecordById",
                            "url": "/appointment/medicalRecord/{id}",
                            "inputBody": null,
                            "methodType": "get",
                            "dependency": null
                        }
                    ]
                },
                {
                    "controllerName": "AppointmentController",
                    "methodName": "setState",
                    "url": "/appointment/setState",
                    "inputBody": "SetStateDTO",
                    "methodType": "post",
                    "dependency": null
                },
            ]
        }
    }
    </microservice relationship>
    You are a software engineer.
    When converting it, it's important to consider the dependencies between services. 
    Additionally, a contract test needs to be created for each microservice.
    The "example microservice relationship" is a dependency relationship within a microservice architecture. 
    It signifies an integration test scenario where the updateConsultation method of the ConsultationController is being tested, 
    relying on functionalities from the AppointmentController (getRecordById method) and the PaymentController (updateConsultationDB method). 
    This involves sending a GET request to getRecordById, including an ID parameter in the path. 
    Additionally, a POST request is made to updateConsultationDB, incorporating an updateDBDTO object in the request.
    Help me convert microservice relationship into a Spring Cloud Contract based on the microservices relationship.
    If there are three microservices in the relationship, three contracts need to be generated.
    Please refer to the project provided in the embedding. When generating contracts, 
    special attention should be paid to ensuring that the generated contracts comply with the logic of long-term transactions in microservices. Additionally, 
    it must be ensured that the contract generated by the previous microservice can be verified by the subsequent ones.
    Print out the Groovy contract file only, without any additional related settings.
    It should include contracts related to testCheckConsultation, getRecordById, getMedicine, and setState.
    Generate contract files with filenames based on the testing objectives of the contracts.
    '''

    find_controller_result = qa({"question": question, "chat_history": chat_history})

    chat_history.append((question, find_controller_result['answer']))
    if os.environ['LANGCHAIN_DEBUG'] == '1':
        print(f"\*\*Question\*\*:\n {question} ")

        print(f"\*\*Answer\*\*:\n {find_controller_result['answer']} ")

def analyze_interface(db):
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 20
    retriever.search_kwargs['k'] = 20
    model = ChatOpenAI(model_name='gpt-4o')  # 'ada' 'gpt-3.5-turbo' 'gpt-4',
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
    project_code, chat_history = analyze_controller(qa)
    result, chat_history = analyze_properties(qa, project_code, chat_history)
    analyze_relationship(qa, result, chat_history)
