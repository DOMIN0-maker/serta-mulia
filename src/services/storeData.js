const { Firestore } = require('@google-cloud/firestore');

async function storeData(id, data, collectionName) {
  const db = new Firestore();

  const predictCollection = db.collection(collectionName);
  return predictCollection.doc(id).set(data);
}

module.exports = storeData;