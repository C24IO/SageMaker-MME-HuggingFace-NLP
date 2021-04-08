import multiprocessing as mp
import numpy as np 
import datetime
import math
import time
import boto3
import botocore
import random
import matplotlib.pyplot as plt

from essential_generators import DocumentGenerator
from sagemaker.serializers import JSONSerializer

sm_client = boto3.client(service_name='sagemaker')
runtime_sm_client = boto3.client(service_name='sagemaker-runtime')

endpoint_name = 'roberta-multimodel-endpoint-v22021-04-07-20-15-52'

def get_result(result):
    global client_times
    client_times.append(result)

def predict(x):
    try:
        target_model = f'roberta-base-{random.randint(0,max_models)}.tar.gz'
        test_data = {"text": 'hello world how are you'}
        jsons = JSONSerializer()
        payload = jsons.serialize(test_data)
        client_start = time.time()
        response = runtime_sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            TargetModel=target_model,
            Body=payload)
        client_end = time.time()
        result = (client_end - client_start)*1000
        return result
    except botocore.exceptions.ClientError as error:
        # Put your error handling logic here
        print(error)
        pass

if __name__ == '__main__':

    # gen = DocumentGenerator()
    # gen.init_word_cache(5000)
    # gen.init_sentence_cache(5000)
    
    total_runs = 5000
    max_models = 50
    errors = 0
    
    client_times = []
    
    content_type = "application/json" 
    accept = "application/octet-stream"
    
    print('Running {} inferences for {} (max models: {}):'.format(total_runs, endpoint_name, max_models))
    
    pool = mp.Pool(mp.cpu_count())
    
    cw_start = datetime.datetime.utcnow()
    
    results = pool.map(predict, tuple(range(0,total_runs,1)))
    
    pool.close()
    pool.join()
    
    for i in results:
        if i == None:
            errors +=1
        else:
            client_times.append(i)
    
    cw_end = datetime.datetime.utcnow()   
    
    cw_duration = cw_end - cw_start 
    
    duration_in_s = cw_duration.total_seconds() 
    
    tps = total_runs/duration_in_s
    
    print('\nErrors - {:.4f} out of {:.4f} total runs | {:.4f}% in {:.4f} seconds \n'.format(errors, total_runs, ((total_runs-errors)/total_runs)*100, duration_in_s))
    
    print('\nTPS: {:.4f}'.format(tps))
        
    print('Client end-to-end latency percentiles:')
    client_avg = np.mean(client_times)
    client_p50 = np.percentile(client_times, 50)
    client_p90 = np.percentile(client_times, 90)
    client_p95 = np.percentile(client_times, 95)
    client_p100 = np.percentile(client_times, 100)
    print('Avg | P50 | P90 | P95 | P100')
    print('{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} \n'.format(client_avg, client_p50, client_p90, client_p95, client_p100))

    
    


