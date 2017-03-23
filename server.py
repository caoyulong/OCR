# this is the server, if we want to test or train in the website, we need to run it.

import BaseHTTPServer
import json
from ocr import OCRNeuralNetwork
import numpy as np
from collections import namedtuple
import csv

HOST_NAME = 'localhost'
PORT_NUMBER = 8000

INPUT_NODE_COUNT = 400
HIDDEN_NODE_COUNT = 49
NUM_OF_TRAINGING = 8
OUTPUT_NODE_COUNT = 10

data_matrix = np.loadtxt(open('mydata.csv', 'rb'), delimiter = ',')
data_labels = np.loadtxt(open('mydataLabels.csv', 'rb'))

data_matrix = data_matrix.tolist()
data_labels = data_labels.tolist()

print 'begin train'
nn = OCRNeuralNetwork(INPUT_NODE_COUNT, HIDDEN_NODE_COUNT, OUTPUT_NODE_COUNT, data_matrix, data_labels, list(range(len(data_matrix))), NUM_OF_TRAINGING);
print 'end train'
nn.save()

class JSONHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_POST(s):
        response_code = 200
        response = ""
        var_len = int(s.headers.get('Content-Length'))
        content = s.rfile.read(var_len);
        payload = json.loads(content);

        if payload.get('train'):
            TrainData = namedtuple('TrainData', ['y0', 'label'])
            td = TrainData(payload['trainArray'][0]['y0'], payload['trainArray'][0]['label'])

            nn.train([td])
            nn.save()
            
            data_file = open('mydata.csv', 'a')
            writer_data = csv.writer(data_file)
            dataLabels_file = open('mydataLabels.csv', 'a')
            writer_dataLabels = csv.writer(dataLabels_file)
            
            writer_data.writerow(td.y0)
            writer_dataLabels.writerow([td.label])

            data_file.close()
            dataLabels_file.close()

        elif payload.get('predict'):
            try:
                response = {"type":"test", "result":nn.predict(str(payload['image']))}
            except:
                response_code = 500
        else:
            response_code = 400

        s.send_response(response_code)
        s.send_header("Content-type", "application/json")
        s.send_header("Access-Control-Allow-Origin", "*")
        s.end_headers()
        if response:
            s.wfile.write(json.dumps(response))
        return

if __name__ == '__main__':
    server_class = BaseHTTPServer.HTTPServer;
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
    else:
        print "Unexpected server exception occurred."
    finally:
        httpd.server_close()
