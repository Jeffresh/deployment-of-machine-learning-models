from flask import Blueprint, request

prediction_app = Blueprint('predicton_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        return 'ok'
