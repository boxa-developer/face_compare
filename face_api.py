import flask
from flask import request, jsonify
import data_builder
from timeit import default_timer

app = flask.Flask(__name__)
app.config['DEBUG'] = True

face_data = [
    'Good'
]

face_rec_object = data_builder.FaceCompare()


@app.route('/', methods=['GET'])
def index():
    print(request.args.get('filename'))
    start = default_timer()
    desc = face_rec_object.compute_descriptor(request.args.get('filename'))
    stop = default_timer()
    data = {
        'data': desc,
        'time_spent': stop - start
    }
    return jsonify(data)


@app.route('/compare/', methods=['GET'])
def compare():
    # face1 = request.args.get('f1')
    face2 = request.args.get('f2')
    start = default_timer()
    f1 = [
      -0.15394993126392365,
      0.009806848131120205,
      0.09327445179224014,
      -0.04600983485579491,
      -0.10328621417284012,
      -0.007000341080129147,
      -0.019526762887835503,
      -0.09647558629512787,
      0.13189418613910675,
      -0.09138338267803192,
      0.2125820368528366,
      0.008694369345903397,
      -0.27849066257476807,
      0.08269702643156052,
      -0.034149568527936935,
      0.1217290535569191,
      -0.1964164525270462,
      -0.09819114953279495,
      -0.1053394079208374,
      -0.08434075862169266,
      -0.07516846805810928,
      0.006826188415288925,
      -0.0011433609761297703,
      0.006965377368032932,
      -0.14048166573047638,
      -0.2899072766304016,
      -0.07303796708583832,
      -0.07131531834602356,
      0.05945860967040062,
      -0.01583717204630375,
      -0.03916701301932335,
      0.05680840089917183,
      -0.1330975890159607,
      -0.044302210211753845,
      0.11341864615678787,
      0.11055823415517807,
      -0.08238356560468674,
      -0.06973264366388321,
      0.3082754611968994,
      -0.020648658275604248,
      -0.20704129338264465,
      0.03516232222318649,
      0.09547913074493408,
      0.29592716693878174,
      0.21581809222698212,
      0.03584398329257965,
      0.047345444560050964,
      -0.1309935450553894,
      0.187177374958992,
      -0.3734530210494995,
      0.027585655450820923,
      0.2090570628643036,
      0.08414351940155029,
      0.0638619214296341,
      0.01482443418353796,
      -0.24296818673610687,
      -0.015241128392517567,
      0.18243125081062317,
      -0.1495916247367859,
      0.07439052313566208,
      0.09104335308074951,
      -0.11656501889228821,
      0.01484163012355566,
      -0.05664665997028351,
      0.2606624662876129,
      0.10655862092971802,
      -0.23150424659252167,
      -0.11026006937026978,
      0.19461175799369812,
      -0.12895730137825012,
      0.014421448111534119,
      0.06224068999290466,
      -0.11976247280836105,
      -0.22375130653381348,
      -0.2257285863161087,
      0.02833346463739872,
      0.396890252828598,
      0.0845210924744606,
      -0.15501686930656433,
      0.02013052999973297,
      -0.18408119678497314,
      -0.05849156528711319,
      0.0052680084481835365,
      0.1495925486087799,
      -0.027470018714666367,
      -0.08789297193288803,
      -0.06793653219938278,
      -0.01787901669740677,
      0.2958393692970276,
      -0.05656662583351135,
      0.07325203716754913,
      0.1948261857032776,
      0.05834590271115303,
      -0.00145728699862957,
      -0.007960911840200424,
      0.14904065430164337,
      -0.23372133076190948,
      -0.0767778679728508,
      -0.17894603312015533,
      -0.009263433516025543,
      -0.046727780252695084,
      -0.054582349956035614,
      0.04268665239214897,
      0.15005657076835632,
      -0.2578778862953186,
      0.26142165064811707,
      -0.045906275510787964,
      -0.06657717376947403,
      -0.0017940225079655647,
      -0.13821035623550415,
      0.04488583654165268,
      0.013822885230183601,
      0.17778868973255157,
      -0.28310084342956543,
      0.19917427003383636,
      0.19481639564037323,
      0.047928813844919205,
      0.19891683757305145,
      0.06779506057500839,
      0.005938908085227013,
      0.009480498731136322,
      0.029612094163894653,
      -0.21354298293590546,
      -0.10174787789583206,
      -0.032506294548511505,
      -0.06867412477731705,
      -0.0009077852591872215,
      0.016087979078292847
    ]
    result = data_builder.compute_similarity(
        f1,
        face_rec_object.compute_descriptor(face2)
    )
    stop = default_timer()
    data = {
        'similarity':result,
        'time':stop-start
    }
    return jsonify(data)


app.run()
