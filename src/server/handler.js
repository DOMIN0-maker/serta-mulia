const predictClassification = require('../services/inferenceService');
const getHistory = require('../services/getHistory');
const storeData = require('../services/storeData');
const crypto = require('crypto');
const collectionName = 'predictions';

async function postPredictHandler(request, h) {
  const { image } = request.payload;
  const { model } = request.server.app;

  const { result, suggestion } = await predictClassification(model, image);
  const id = crypto.randomUUID();
  const createdAt = new Date().toISOString();

  const data = {
    "id": id,
    "result": result,
    "suggestion": suggestion,
    "createdAt": createdAt
  }

  await storeData(id, data, collectionName);

  const response = h.response({
    status: 'success',
    message: 'Model is predicted successfully',
    data
  })
  response.code(201);
  return response;
}

async function getHistoryHandler(request, h) {
  const data = await getHistory(collectionName);

  const response = h.response({
    status: 'success',
    data: data
  })
  response.code(200);
  return response;
}

module.exports = { postPredictHandler, getHistoryHandler };