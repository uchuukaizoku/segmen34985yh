import * as tf from "@tensorflow/tfjs-core";
import {
  getLabels,
  getColormap,
  toSegmentationImage
} from "@tensorflow-models/deeplab";

const deeplab = require("@tensorflow-models/deeplab");

const loadModel = async () => {
  // const tf = require("@tensorflow/tfjs-core");
  // const deeplab = require("@tensorflow-models/deeplab");
  // const tf = require("@tensorflow/tfjs-core");

  const modelName = "cityscapes"; // set to your preferred model, either `pascal`, `cityscapes` or `ade20k`
  const quantizationBytes = 4; // either 1, 2 or 4
  return await deeplab.load({ base: modelName, quantizationBytes });
  // return await deeplab.load();
};

(async () => {
  const img = document.getElementById("img");

  // const floatImageTensor = tf.cast(imageTensor, 'float32');

  console.time("loadModel");
  const model = await loadModel();
  console.timeEnd("loadModel");

  console.time("segment");
  const segments = await model.segment(img);
  console.timeEnd("segment");
  console.log(segments.legend);

  const getSemanticSegmentationMap = (image) => {
    return model.predict(image);
  };

  const maria = await getSemanticSegmentationMap(img);
  console.log("maria");
  //console.log(maria);

  const canvas = document.getElementById("canvas");
  const translateSegmentationMap = async (segmentationMap) => {
    return await toSegmentationImage(
      getColormap("cityscapes"),
      getLabels("cityscapes"),
      segmentationMap,
      canvas
    );
  };

  const translate = await translateSegmentationMap(maria);

  return;
})();

const loadImage = async (path) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onerror = (err) => reject(err);
    img.onload = () => resolve(img);
    img.src = path;
  });
};
