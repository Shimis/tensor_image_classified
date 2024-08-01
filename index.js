const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const fs = require('fs');

// Загрузите изображение
const image = fs.readFileSync('./images/test_5.jpeg');
const decodedImage = tf.node.decodeImage(image, 3);

(async () => {
  // Загрузите модель MobileNet
  const model = await mobilenet.load({version: 2, alpha: 1.0});

  // Выполните предсказание
  const predictions = await model.classify(decodedImage);

  console.log('mobilenet Predictions: ');
  console.log(predictions);

})();

(async () => {
    // Загрузите модель Coco-SSD
    const model = await cocoSsd.load();
  
    // Выполните предсказание
    const predictions = await model.detect(decodedImage);
  
    console.log('cocoSsd Predictions: ');
    console.log(predictions);
  })();