from json import JSONDecodeError

from coverage.jsonreport import JsonObj
from flask import Flask



app = Flask(__name__)

@app.route('/calibrate', methods=['GET'])
def calibrate():
    """
    キャリブレーションを行うためのエンドポイント
    """
    raise JSONDecodeError
    pass

@app.route('/quite', methods=['GET'])
def quite():
    pass
    """
    アプリケーションの終了を行うためのエンドポイント
    """