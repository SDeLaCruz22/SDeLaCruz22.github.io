//import 'babel-polyfill'
import * as tf from '@tensorflow/tfjs';
import {TestWeb} from '/testWeb';
import imageURL from './sd2.jpg';


const sd2 = document.getElementById('sd2');
sd2.onload = async () => {
  const resultElement = document.getElementById('result');

  resultElement.innerText = 'Loading testWeb...';

  const testWeb = new TestWeb();
  console.time('Loading of model');
  await testWeb.load();
  console.timeEnd('Loading of model');

  const pixels = tf.browser.fromPixels(sd2);

  console.time('First prediction');
  let result = testWeb.predict(pixels);
  console.timeEnd('First prediction');

  /*
  resultElement.innerText = '';
  topK.forEach(x => {
    resultElement.innerText += `${x.value.toFixed(3)}: ${x.label}\n`;
  });

  console.time('Subsequent predictions');
  result = testWeb.predict(pixels);
  testWeb.getTopKClasses(result, 5);
  console.timeEnd('Subsequent predictions');
*/
  testWeb.dispose();
};
sd2.src = imageURL;