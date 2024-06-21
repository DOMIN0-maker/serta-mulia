const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat()

        const classes = ['Melanocytic nevus', 'Squamous cell carcinoma', 'Vascular lesion'];

        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;

        const classResult = tf.argMax(prediction, 1).dataSync()[0];
        const label = classes[classResult];

        let result, suggestion;


        switch (label) {
            case 'Melanocytic nevus':
                result = "Cancer"
                suggestion = "dokter abdur berkata: Segera konsultasi dengan dokter terdekat jika ukuran semakin membesar dengan cepat, mudah luka atau berdarah."
                break;
            case 'Squamous cell carcinoma':
                result = "Cancer"
                suggestion = "dokter abdur berkata: Segera konsultasi dengan dokter terdekat untuk meminimalisasi penyebaran kanker."
                break;
            case 'Vascular lesion':
                result = "Cancer"
                suggestion = "dokter abdur berkata: Segera konsultasi dengan dokter terdekat untuk mengetahui detail terkait tingkat bahaya penyakit."
                break;
            default:
                result = "Non-Cancer"
                suggestion = "Anda sehat kata dokter abdur!"
                break;
        }

        return { confidenceScore, label, result, suggestion };
    } catch (error) {
        throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`)
    }
}

module.exports = predictClassification;