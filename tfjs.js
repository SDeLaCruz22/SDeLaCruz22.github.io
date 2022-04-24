import * as tf from '@tensorflow/tfjs';

const JSONFOLDER = './tfjsmodel';
const MODELFILE = 'model.json';

const PREPROCESSDIVISOR = tf.scalar(255);

export class TestWeb{
    constructor(){}

    async load() {
        this.model = await tf.loadGraphModel(
            JSONFOLDER + MODELFILE
        );
    }

    dispose() {
        if(this.model){
            this.model.dispose();
        }
    }
}

predict(input){
    const preprocessed = tf.div(
        tf.sub(input.asType('float32'), PREPROCESSDIVISOR),
        PREPROCESSDIVISOR
    );
    const reshaped = preprocessed.reshape([1, ..preprocessed.shape]);
    
    return this.model.execute(
        {['images']: reshaped}, './testWeb'
    )
}