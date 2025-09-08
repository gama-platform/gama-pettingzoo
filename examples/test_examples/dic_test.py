import json
import asyncio
from pathlib import Path

from gama_client.sync_client import GamaSyncClient
from gama_client.message_types import MessageTypes


async def main():
    
    client = GamaSyncClient("localhost", 1001)

    # Set up experiment parameters
    exp_path = str(Path(__file__).parents[0] / "dic_model.gaml")  # Path to the GAMA model
    exp_name = "dic"  # Name of the experiment to run
    
    print("Connecting to GAMA server")
    try:
        client.connect()
    except Exception as e:
        print("Error while connecting to the server:", e)
        return
    
    print("Loading GAML model")
    gama_response = client.load(exp_path, exp_name, False, False, False, True)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while loading model:", gama_response)
        return
    
    print("Initialization successful")
    experiment_id = gama_response["content"]
    
    # st = "Hello, GAMA!"
    st = "{\"key\": value}"
    dic = {"key":123.4}
    
    print("Sending string value")
    expr = f"st<-\'{st}\';"  # GAMA expression to set seed to 0
    gama_response = client.expression(experiment_id, expr)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while sending string:", gama_response)
        return
    print("string value set successfully")
    
    print("Getting current string value")
    gama_response = client.expression(experiment_id, r"st")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while getting string:", gama_response)
        return
    
    print("content:", gama_response["content"])
    print("content type:", type(gama_response["content"]))
    # st_value = json.loads(f"{gama_response['content']}")
    # print("Current string value:", st_value)

    print("Sending dictionary value")
    expr = f"dic<-from_json(\'{json.dumps(dic)}\');"  # GAMA expression to set seed to 0
    gama_response = client.expression(experiment_id, expr)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while sending dictionary:", gama_response)
        return
    print("dictionary value set successfully")

    print("Getting current dictionary value")
    gama_response = client.expression(experiment_id, r"dic")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while getting dictionary:", gama_response)
        return
    # Parse the JSON response to get the actual value
    print("content:", gama_response["content"])
    print("content type:", type(gama_response["content"]))
    # dic_value = json.loads(f"{gama_response['content']}")
    # print("Current dictionary value:", dic_value)

    print("Closing connection to GAMA server")
    client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())