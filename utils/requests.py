from time import time, sleep

import requests, json

def __make_infomation(
        origin: list[float],
        destination: list[float],
        response: dict,
        routing_mode: str
) -> dict:
    route = response["route"][routing_mode][0]
    summary = route["summary"]
    return {
        "start": origin,
        "goal": destination,
        "distance": summary["distance"],
        "duration": summary["duration"],
        "fuelPrice": summary["fuelPrice"],
        "tollFare": summary["tollFare"],
        "path": route["path"]
    }


def __make_params(origin: list[float], destination: list[float], routing_mode: str):
    return {
        "start": f"{origin[1]},{origin[0]}",
        "goal": f"{destination[1]},{destination[0]}",
        "option": routing_mode,
        "cartype": 3,
        "fueltype": "diesel",
        "mileage": 7,
    }


def __routing_error(
        i: int, j: int,
        origin: list[float],
        destination: list[float],
        routing_mode: str,
        response: dict,
):
    error_message = \
        f'{response["message"]}\n' + \
        "Error Occured while find route between these two points :\n" + \
        f'({", ".join(map(str, origin))}) -> ({", ".join(map(str, destination))}) | {routing_mode = }\n'
    
    print(error_message)
    return {
        "i": i, "j": j,
        "start": origin, "goal": destination,
        "routing_mode": routing_mode
    }


def make_dmat_by_coordinates(
        datafile_name: str = './data/bin.json',
        outputfile_name: str = './data/dmat.json',
        errorfile_name: str = './data/error.json',
        encoding: str = 'utf-8',
        routing_mode: str = 'trafast'
):
    API_KEY = json.loads(open('./data/api_key.json', encoding='utf-8').read())
    API_header_1 = "x-ncp-apigw-api-key-id"
    API_header_2 = "x-ncp-apigw-api-key"

    path = 'https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving'
    headers = {
        API_header_1: API_KEY[API_header_1],
        API_header_2: API_KEY[API_header_2]
    }

    bins = json.loads(open(datafile_name, encoding=encoding).read())
    n = len(bins)

    base_response = {
        "code": 0,
        "route": {
            routing_mode: [
                {
                    "path": [],
                    "summary": {
                        "distance": 0,
                        "duration": 0,
                        "fuelPrice": 0,
                        "tollFare": 0
                    }
                }
            ]
        }
    }

    distance_matrix = json.loads(open(outputfile_name, encoding=encoding).read())
    error = []
    startTime = time()
    try:
        for i, origin in enumerate(bins):
            for j, destination in enumerate(bins):
                requestStartTime = time()
                if i == j:
                    response = base_response
                else:
                    iter = 1
                    while True:
                        try:
                            response = requests.get(
                                path,
                                headers=headers,
                                params=__make_params(origin, destination, routing_mode)
                            ).json()
                            
                            break
                        except requests.exceptions.ConnectionError:
                            sleep(10 * i)
                            iter += 1
                            if iter > 10:
                                print(f'Current (row, col) = ({i}, {j}), start from it later.')
                                raise Exception('fuck')

                if response["code"] != 0:
                    error.append(__routing_error(i, j, origin, destination, routing_mode, response))
                    continue

                distance_matrix[i][j] = __make_infomation(
                    origin,
                    destination,
                    response,
                    routing_mode
                )
                
                print(f'd({i},{j}) completed, {time() - requestStartTime:.4f}s elapsed, total {time() - startTime:.4f}s elapsed.')
    finally:
        json.dump(distance_matrix, fp=open(outputfile_name, 'w', encoding=encoding), ensure_ascii=False)
        if len(error) > 0:
            json.dump(error, fp=open(errorfile_name, 'w', encoding=encoding), ensure_ascii=False)
