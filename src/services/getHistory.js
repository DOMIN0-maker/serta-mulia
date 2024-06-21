const { Firestore } = require('@google-cloud/firestore');

async function getHistory(collectionName) {
  const db = new Firestore();
  let data = [];

  const historiesCollection = db.collection(collectionName);
  const dataCollection = await historiesCollection.get();

  dataCollection.forEach(doc => {
    data.push({
      id: doc.id,
      ...doc.data()
    });
  });

  return data;
}

module.exports = getHistory;