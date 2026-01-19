
// =============================================================================
// MAIN APPLICATION LOGIC
// =============================================================================

document.addEventListener('DOMContentLoaded', async () => {

    // --- Global State ---
    let model = null;
    let lossChart = null;
    let workspace = null;
    let classNames = []; // Will be populated from the dataset folder structure
    let datasetFiles = { images: [], labels: [] };
    let isDirty = false; // Track unsaved changes
    let trainingHistory = { epochs: [], losses: [] }; // 学習履歴を保存

    // --- UI Elements ---
    const statusSpan = document.getElementById('status-span');
    const progressBar = document.getElementById('progress-bar');
    const trainButton = document.getElementById('train-button');
    const downloadModelButton = document.getElementById('download-model-button');
    const uploadModelInput = document.getElementById('upload-model-input');
    const inferButton = document.getElementById('infer-button');
    const inferenceImageInput = document.getElementById('inference-image-input');
    const inferenceImagePreview = document.getElementById('inference-image-preview');
    const predictionResult = document.getElementById('prediction-result');
    const datasetFolderInput = document.getElementById('dataset-folder-input');
    const epochsInput = document.getElementById('epochs-input');
    const batchSizeInput = document.getElementById('batch-size-input');

    // --- Main Initialization ---
    await initializeTfjs();
    initializeBlockly();
    initializeChart();
    setupEventListeners();

    // --- Initialization Functions ---
    async function initializeTfjs() {
        try {
            await tf.setBackend('webgpu');
            console.log('Using WebGPU backend.');
            updateStatus('Ready (WebGPU backend)');
        } catch (e) {
            console.warn('WebGPU is not available. Falling back to default backend.');
            updateStatus('Ready (WebGL/CPU backend)');
        }
    }

    function initializeBlockly() {
        const toolboxXml = `
        <xml id="toolbox" style="display: none">
            <category name="Model" colour="230">
                <block type="model_builder"></block>
                <block type="model_compiler"></block>
                <block type="optimizer_adam"></block>
            </category>
            <category name="Layers" colour="210">
                <block type="layer_input"></block>
                <block type="layer_conv2d"></block>
                <block type="layer_maxpooling2d"></block>
                <block type="layer_flatten"></block>
                <block type="layer_dense"></block>
                <block type="layer_dropout"></block>
            </category>
        </xml>`;
        workspace = Blockly.inject('blocklyDiv', {
            toolbox: toolboxXml,
            scrollbars: true,
            trashcan: true,
            renderer: 'zelos',
            zoom: {
                controls: true,
                wheel: true, // Requires Ctrl key
                startScale: 1.0,
                maxScale: 3,
                minScale: 0.3,
                scaleSpeed: 1.2,
                pinch: true
            },
            move: {
                scrollbars: {
                    vertical: true,
                    horizontal: true
                },
                drag: true,
                wheel: true
            }
        });

        // Set the dirty flag on any substantive change
        workspace.addChangeListener((event) => {
            // Don't flag UI events like zoom or drag as changes
            if (event.isUiEvent) {
                return;
            }
            isDirty = true;
        });
    }

    function initializeChart() {
        const ctx = document.getElementById('loss-chart').getContext('2d');
        lossChart = new Chart(ctx, { type: 'line', data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: 'rgba(255, 99, 132, 1)', borderWidth: 2, fill: false }] }, options: { scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'Loss' } } }, animation: { duration: 0 } } });
    }

    function setupEventListeners() {
        trainButton.addEventListener('click', handleTrain);
        inferButton.addEventListener('click', handleInfer);
        downloadModelButton.addEventListener('click', handleDownloadModel);
        uploadModelInput.addEventListener('change', handleUploadModel);
        inferenceImageInput.addEventListener('change', handleInferenceImageSelect);
        datasetFolderInput.addEventListener('change', handleDatasetSelection);

        // Warn user before leaving if there are unsaved changes
        window.addEventListener('beforeunload', (event) => {
            if (isDirty) {
                event.preventDefault();
                // Most modern browsers show a generic message, but this is required.
                event.returnValue = '';
            }
        });
    }

    // --- Helper Functions ---
    function updateStatus(message, isBusy = false) { statusSpan.textContent = message; trainButton.disabled = isBusy; inferButton.disabled = isBusy; }
    function updateProgressBar(epoch, totalEpochs) { const percentage = totalEpochs > 0 ? Math.round((epoch / totalEpochs) * 100) : 0; progressBar.style.width = `${percentage}%`; progressBar.textContent = `${percentage}%`; progressBar.setAttribute('aria-valuenow', percentage); }
    function resetChart() { 
        trainingHistory = { epochs: [], losses: [] };
        lossChart.data.labels = []; 
        lossChart.data.datasets[0].data = []; 
        lossChart.update(); 
    }
    function restoreChart() {
        lossChart.data.labels = trainingHistory.epochs;
        lossChart.data.datasets[0].data = trainingHistory.losses;
        lossChart.update();
    }

    // =============================================================================
    // BLOCK PARSING & RECONSTRUCTION
    // =============================================================================
    function generateConfigFromBlocks() {
        try {
            const topBlocks = workspace.getTopBlocks(true);
            const modelBlock = topBlocks.find(b => b.type === 'model_builder');

            if (!modelBlock) throw new Error('Could not find a "Sequential Model" block.');

            // --- Parse Model Config ---
            const modelConfig = { layers: [] };
            let inputShape = null; // Local to this function
            let layerBlock = modelBlock.getInput('LAYERS').connection.targetBlock();
            while (layerBlock) {
                switch (layerBlock.type) {
                    case 'layer_input':
                        const shape = {
                            height: parseInt(layerBlock.getFieldValue('HEIGHT'), 10),
                            width: parseInt(layerBlock.getFieldValue('WIDTH'), 10),
                            channels: parseInt(layerBlock.getFieldValue('CHANNELS'), 10)
                        };
                        modelConfig.layers.push({ type: 'input', shape: [shape.height, shape.width, shape.channels] });
                        inputShape = shape; // Set local inputShape
                        break;
                    case 'layer_dense':
                        modelConfig.layers.push({ type: 'dense', units: parseInt(layerBlock.getFieldValue('UNITS'), 10), activation: layerBlock.getFieldValue('ACTIVATION') });
                        break;
                    case 'layer_conv2d':
                        modelConfig.layers.push({ type: 'conv2d', filters: parseInt(layerBlock.getFieldValue('FILTERS'), 10), kernelSize: parseInt(layerBlock.getFieldValue('KERNEL_SIZE'), 10), activation: layerBlock.getFieldValue('ACTIVATION') });
                        break;
                    case 'layer_maxpooling2d':
                        modelConfig.layers.push({ type: 'maxPooling2d', poolSize: parseInt(layerBlock.getFieldValue('POOL_SIZE'), 10) });
                        break;
                    case 'layer_flatten':
                        modelConfig.layers.push({ type: 'flatten' });
                        break;
                    case 'layer_dropout':
                        modelConfig.layers.push({ type: 'dropout', rate: parseFloat(layerBlock.getFieldValue('RATE')) });
                        break;
                }
                layerBlock = layerBlock.getNextBlock();
            }
            if (!inputShape) throw new Error('An "Input Layer" block is required.');

            // --- Parse Compiler Config ---
            const compilerBlock = modelBlock.getInput('COMPILER').connection.targetBlock();
            if (!compilerBlock) {
                // Return null for compiler if it's not connected, instead of throwing an error
                return { model: modelConfig, compiler: null, inputShape: inputShape };
            }

            const loss = compilerBlock.getFieldValue('LOSS');
            const optimizerBlock = compilerBlock.getInput('OPTIMIZER').connection.targetBlock();
            if (!optimizerBlock) throw new Error('Optimizer block is not connected.');
            
            const optimizerConfig = {
                type: optimizerBlock.type.replace('optimizer_', ''),
                learningRate: parseFloat(optimizerBlock.getFieldValue('LEARNING_RATE'))
            };
            const compilerConfig = { optimizer: optimizerConfig, loss: loss };

            return { model: modelConfig, compiler: compilerConfig, inputShape: inputShape };

        } catch (e) {
            alert(`Error reading blocks: ${e.message}`);
            console.error(e);
            return null;
        }
    }

    function rebuildBlocksFromTopology(parsedJson) {
        const topology = parsedJson.modelTopology; // Changed from parsedJson
        const trainingConfig = parsedJson.trainingConfig; // Changed from parsedJson

        if (!topology || !topology.config || !topology.config.layers) {
            console.error('Invalid model topology provided for block reconstruction.');
            return;
        }

        workspace.clear();

        const modelBlock = workspace.newBlock('model_builder');
        modelBlock.initSvg();
        modelBlock.render();

        let lastLayerConnection = modelBlock.getInput('LAYERS').connection;

        const layers = topology.config.layers;
        for (const layerConfig of layers) {
            const className = layerConfig.class_name;
            const config = layerConfig.config;
            let blockType = null;

            switch (className) {
                case 'InputLayer':   blockType = 'layer_input'; break;
                case 'Conv2D':       blockType = 'layer_conv2d'; break;
                case 'MaxPooling2D': blockType = 'layer_maxpooling2d'; break;
                case 'Flatten':      blockType = 'layer_flatten'; break;
                case 'Dense':        blockType = 'layer_dense'; break;
                case 'Dropout':      blockType = 'layer_dropout'; break;
            }

            if (blockType) {
                const newBlock = workspace.newBlock(blockType);
                switch (blockType) {
                    case 'layer_input':
                        if (config.batch_input_shape && config.batch_input_shape.length === 4) {
                            newBlock.setFieldValue(config.batch_input_shape[2], 'WIDTH');
                            newBlock.setFieldValue(config.batch_input_shape[1], 'HEIGHT');
                            newBlock.setFieldValue(String(config.batch_input_shape[3]), 'CHANNELS');
                        }
                        break;
                    case 'layer_dense':
                        newBlock.setFieldValue(config.units, 'UNITS');
                        newBlock.setFieldValue(config.activation, 'ACTIVATION');
                        break;
                    case 'layer_conv2d':
                        newBlock.setFieldValue(config.filters, 'FILTERS');
                        newBlock.setFieldValue(config.kernel_size[0], 'KERNEL_SIZE');
                        newBlock.setFieldValue(config.activation, 'ACTIVATION');
                        break;
                    case 'layer_maxpooling2d':
                        newBlock.setFieldValue(config.pool_size[0], 'POOL_SIZE');
                        break;
                    case 'layer_dropout':
                        newBlock.setFieldValue(config.rate, 'RATE');
                        break;
                }
                newBlock.initSvg();
                newBlock.render();
                lastLayerConnection.connect(newBlock.previousConnection);
                lastLayerConnection = newBlock.nextConnection;
            }
        }

        if (trainingConfig && trainingConfig.optimizer_config) {
            const compilerBlock = workspace.newBlock('model_compiler');
            if (trainingConfig.loss) {
                compilerBlock.setFieldValue(trainingConfig.loss, 'LOSS');
            }
            const optimizerConfig = trainingConfig.optimizer_config;
            if (optimizerConfig.class_name === 'Adam') {
                const optimizerBlock = workspace.newBlock('optimizer_adam');
                if (optimizerConfig.config && typeof optimizerConfig.config.learning_rate !== 'undefined') {
                    optimizerBlock.setFieldValue(optimizerConfig.config.learning_rate, 'LEARNING_RATE');
                }
                optimizerBlock.initSvg();
                optimizerBlock.render();
                compilerBlock.getInput('OPTIMIZER').connection.connect(optimizerBlock.outputConnection);
            }
            compilerBlock.initSvg();
            compilerBlock.render();
            modelBlock.getInput('COMPILER').connection.connect(compilerBlock.previousConnection);
        }
    }

    // =============================================================================
    // DATASET HANDLING
    // =============================================================================
    async function handleDatasetSelection(event) {
        updateStatus('Loading dataset... ', true);
        try {
            const files = Array.from(event.target.files);
            if (files.length === 0) throw new Error("No folder selected or folder is empty.");

            const tempClassNames = new Set();
            const imageData = [];

            for (const file of files) {
                const pathParts = file.webkitRelativePath.split('/');
                if (pathParts.length > 2) {
                    const className = pathParts[pathParts.length - 2];
                    tempClassNames.add(className);
                    imageData.push({ file, className });
                }
            }

            if (imageData.length === 0) {
                throw new Error("No valid image files found. Please check the folder structure.");
            }

            classNames = [...tempClassNames].sort();
            const classMap = new Map(classNames.map((name, index) => [name, index]));

            datasetFiles.images = imageData.map(d => d.file);
            datasetFiles.labels = imageData.map(d => classMap.get(d.className));

            updateStatus(`${datasetFiles.images.length} images loaded for ${classNames.length} classes. Ready to train.`, false);
            console.log('Classes found:', classNames);

        } catch (error) {
            alert(`Error loading dataset: ${error.message}`);
            console.error(error);
            updateStatus('Error loading dataset.', false);
        }
    }

    // =============================================================================
    // MODEL TRAINING
    // =============================================================================
    async function handleTrain() {
        updateStatus('Starting training...', true);
        // チャートはリセットしない（追加学習の場合は続きから）
        updateProgressBar(0, 1);
        try {
            const config = generateConfigFromBlocks();
            if (!config) throw new Error("Configuration could not be generated from blocks.");
            if (!config.compiler) throw new Error("Compiler settings not found. Please connect the compiler blocks.");
            if (datasetFiles.images.length === 0) throw new Error("Dataset not loaded. Please select a dataset folder first.");
            
            const numClasses = classNames.length;
            if (numClasses < 2) throw new Error("Dataset must contain at least 2 classes.");

            const epochs = parseInt(epochsInput.value, 10);
            const batchSize = parseInt(batchSizeInput.value, 10);
            
            updateStatus('Preparing data...');
            const { inputShape } = config;

            const dataset = tf.data.generator(function*() {
                for (let i = 0; i < datasetFiles.images.length; i++) {
                    yield { imageFile: datasetFiles.images[i], labelIndex: datasetFiles.labels[i] };
                }
            }).shuffle(datasetFiles.images.length).mapAsync(async ({ imageFile, labelIndex }) => {
                const img = await loadImage(imageFile);
                const tensor = tf.tidy(() => {
                    let tempTensor = tf.browser.fromPixels(img);
                    if (inputShape.channels === 1 && tempTensor.shape[2] === 3) {
                        tempTensor = tempTensor.mean(2).expandDims(2);
                    }
                    return tempTensor.resizeBilinear([inputShape.height, inputShape.width])
                                     .toFloat()
                                     .div(tf.scalar(255.0));
                });
                const oneHotLabel = tf.oneHot(labelIndex, numClasses);
                return { xs: tensor, ys: oneHotLabel };
            }).batch(batchSize);

            function loadImage(file) { 
                return new Promise((resolve, reject) => { 
                    const reader = new FileReader(); 
                    reader.onload = (e) => { 
                        const img = new Image(); 
                        img.onload = () => resolve(img); 
                        img.onerror = reject; 
                        img.src = e.target.result; 
                    }; 
                    reader.onerror = reject; 
                    reader.readAsDataURL(file); 
                }); 
            }

            // モデルが既に存在するかチェック
            const isRetraining = model !== null;
            const startEpoch = trainingHistory.epochs.length;
            
            if (!isRetraining) {
                updateStatus('Building model...');
                model = tf.sequential();
                model.add(tf.layers.inputLayer({ inputShape: [inputShape.height, inputShape.width, inputShape.channels] }));
                
                for (const layerConfig of config.model.layers) {
                    if (layerConfig.type === 'input') continue;
                    const layer = createLayer(layerConfig);
                    if (layer) model.add(layer);
                }

                const lastLayer = model.layers[model.layers.length - 1];
                if (lastLayer.units !== numClasses) {
                     console.warn(`The last layer's units (${lastLayer.units}) do not match the number of classes (${numClasses}). Adjusting the final layer for classification.`);
                     model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
                }

                const optimizer = createOptimizer(config.compiler.optimizer);
                model.compile({ optimizer: optimizer, loss: config.compiler.loss, metrics: ['accuracy'] });
                model.summary();
            } else {
                updateStatus('Continuing training from previous state...');
                console.log(`Resuming from epoch ${startEpoch}`);
            }

            updateStatus('Training...');
            await model.fitDataset(dataset, { 
                epochs: epochs, 
                callbacks: { 
                    onEpochEnd: (epoch, logs) => { 
                        const globalEpoch = startEpoch + epoch + 1;
                        console.log(`Epoch ${globalEpoch} - Loss: ${logs.loss.toFixed(4)}`); 
                        trainingHistory.epochs.push(globalEpoch);
                        trainingHistory.losses.push(logs.loss);
                        lossChart.data.labels.push(globalEpoch); 
                        lossChart.data.datasets[0].data.push(logs.loss); 
                        lossChart.update(); 
                        updateProgressBar(epoch + 1, epochs); 
                    } 
                }
            });

            updateStatus('Training complete! Ready for inference.', false);
        } catch (error) {
            alert(`Training failed: ${error.message}`);
            console.error(error);
            updateStatus('Training failed. Please check console.', false);
        }
    }

    function createLayer(config) {
        switch (config.type) {
            case 'dense': return tf.layers.dense({ units: config.units, activation: config.activation });
            case 'conv2d': return tf.layers.conv2d({ filters: config.filters, kernelSize: config.kernelSize, activation: config.activation });
            case 'maxPooling2d': return tf.layers.maxPooling2d({ poolSize: config.poolSize });
            case 'flatten': return tf.layers.flatten();
            case 'dropout': return tf.layers.dropout({ rate: config.rate });
            default: console.warn(`Unknown layer type: ${config.type}`); return null;
        }
    }

    function createOptimizer(config) {
        switch (config.type) {
            case 'adam': return tf.train.adam(config.learningRate); 
            default: console.warn(`Unknown optimizer type: ${config.type}. Using default Adam.`); return tf.train.adam();
        }
    }

    // =============================================================================
    // INFERENCE & MODEL MANAGEMENT
    // =============================================================================

    async function handleInfer() {
        updateStatus('Running inference...', true);
        let tensor;
        try {
            if (!model) throw new Error('Model is not available. Please train or upload a model first.');
            if (inferenceImageInput.files.length === 0) throw new Error('Please select an image to run inference on.');
            if (classNames.length === 0) throw new Error('Class names are not defined. Please load a dataset or a model with embedded class names.');

            const config = generateConfigFromBlocks();
            if (!config || !config.inputShape) {
                throw new Error("Could not determine input shape from the blocks. Is an Input Layer block present?");
            }
            const currentInputShape = config.inputShape;

            const imageElement = inferenceImagePreview;
            tensor = tf.tidy(() => {
                let tempTensor = tf.browser.fromPixels(imageElement);
                if (currentInputShape.channels === 1 && tempTensor.shape[2] === 3) {
                    tempTensor = tempTensor.mean(2).expandDims(2);
                }
                return tempTensor.resizeBilinear([currentInputShape.height, currentInputShape.width])
                                 .toFloat()
                                 .div(tf.scalar(255.0))
                                 .expandDims(0);
            });

            const prediction = model.predict(tensor);
            const probabilities = await prediction.data();
            const predictedIndex = prediction.argMax(-1).dataSync()[0];
            const predictedClass = classNames[predictedIndex];
            const confidence = probabilities[predictedIndex] * 100;

            predictionResult.textContent = `${predictedClass} (${confidence.toFixed(2)}% confidence)`;
            updateStatus('Inference complete.', false);
        } catch (error) {
            alert(`Inference failed: ${error.message}`);
            console.error(error);
            updateStatus('Inference failed.', false);
        } finally {
            if (tensor) tensor.dispose();
        }
    }

    function handleInferenceImageSelect(event) {
        const file = event.target.files[0];
        if (file) { const reader = new FileReader(); reader.onload = (e) => { inferenceImagePreview.src = e.target.result; inferenceImagePreview.style.display = 'block'; predictionResult.textContent = ''; }; reader.readAsDataURL(file); }
    }

    async function handleDownloadModel() {
        updateStatus('Saving model...', true);
        try {
            if (!model) throw new Error('No model to save. Please train a model first.');
            if (classNames.length === 0) throw new Error('Cannot save a model without class names. Please train the model on a dataset first.');

            const config = generateConfigFromBlocks();
            if (!config) throw new Error("Could not generate config from blocks for saving.");

            // メモリ内のIOHandlerに一時保存
            const saveResult = await model.save(tf.io.withSaveHandler(async (artifacts) => {
                return artifacts;
            }));

            // カスタムメタデータとトレーニング設定を追加してmodel.jsonを構築
            const modelJson = {
                modelTopology: saveResult.modelTopology,
                weightsManifest: [{
                    paths: ['weights.bin'],
                    weights: saveResult.weightSpecs
                }],
                format: saveResult.format,
                generatedBy: saveResult.generatedBy,
                convertedBy: saveResult.convertedBy,
                userDefinedMetadata: { 
                    classNames,
                    trainingHistory: {
                        epochs: trainingHistory.epochs,
                        losses: trainingHistory.losses
                    }
                }
            };

            // トレーニング設定を追加
            if (config.compiler) {
                modelJson.trainingConfig = {
                    loss: config.compiler.loss,
                    metrics: ['accuracy'],
                    optimizer_config: {
                        class_name: config.compiler.optimizer.type.charAt(0).toUpperCase() + config.compiler.optimizer.type.slice(1),
                        config: { learning_rate: config.compiler.optimizer.learningRate }
                    }
                };
            }
            
            // JSONファイルをダウンロード
            const modelBlob = new Blob([JSON.stringify(modelJson, null, 2)], { type: 'application/json' });
            const modelUrl = URL.createObjectURL(modelBlob);
            const aModel = document.createElement('a');
            aModel.href = modelUrl;
            aModel.download = 'model.json';
            document.body.appendChild(aModel);
            aModel.click();
            document.body.removeChild(aModel);
            URL.revokeObjectURL(modelUrl);
            
            // バイナリファイルをダウンロード
            const weightsBlob = new Blob([saveResult.weightData], { type: 'application/octet-stream' });
            const weightsUrl = URL.createObjectURL(weightsBlob);
            const aWeights = document.createElement('a');
            aWeights.href = weightsUrl;
            aWeights.download = 'weights.bin';
            document.body.appendChild(aWeights);
            aWeights.click();
            document.body.removeChild(aWeights);
            URL.revokeObjectURL(weightsUrl);

            updateStatus('Model saved as model.json and weights.bin.', false);
            isDirty = false;

        } catch (error) {
            alert(`Failed to save model: ${error.message}`);
            console.error(error);
            updateStatus('Failed to save model.', false);
        }
    }

    async function handleUploadModel(event) {
        updateStatus('Loading model...', true);
        try {
            const files = Array.from(event.target.files);
            console.log('Selected files:', files.map(f => ({ name: f.name, size: f.size, type: f.type })));
            
            if (files.length !== 2) throw new Error('Please select both the model JSON and weights BIN files.');
            
            // ファイルを名前で正確に識別
            let jsonFile = files.find(f => f.name === 'model.json');
            let weightsFile = files.find(f => f.name === 'weights.bin');
            
            // 見つからない場合は拡張子で探す
            if (!jsonFile) jsonFile = files.find(f => f.name.endsWith('.json'));
            if (!weightsFile) weightsFile = files.find(f => f.name.endsWith('.bin'));

            if (!jsonFile || !weightsFile) throw new Error('Both a .json and a .bin file are required.');
            
            console.log('JSON file:', jsonFile.name, jsonFile.size);
            console.log('Weights file:', weightsFile.name, weightsFile.size);
            
            const jsonText = await jsonFile.text();
            const parsedJson = JSON.parse(jsonText);
            console.log('Parsed JSON successfully');
            
            // ブロックを再構築
            rebuildBlocksFromTopology(parsedJson);
            console.log('Blocks rebuilt successfully');

            // カスタムIOHandlerを使ってモデルをロード
            console.log('Loading model with custom handler...');
            
            const weightsArrayBuffer = await weightsFile.arrayBuffer();
            console.log('Weights loaded, size:', weightsArrayBuffer.byteLength);
            
            // trainingConfigを除外してモデルをロード（後で手動でコンパイル）
            model = await tf.loadLayersModel({
                load: async () => {
                    return {
                        modelTopology: parsedJson.modelTopology,
                        weightSpecs: parsedJson.weightsManifest[0].weights,
                        weightData: weightsArrayBuffer,
                        format: parsedJson.format,
                        generatedBy: parsedJson.generatedBy,
                        convertedBy: parsedJson.convertedBy,
                        userDefinedMetadata: parsedJson.userDefinedMetadata
                        // trainingConfigは意図的に除外
                    };
                }
            });
            
            console.log('Model loaded successfully');
            
            // モデルをコンパイル（必須）
            if (parsedJson.trainingConfig && parsedJson.trainingConfig.optimizer_config) {
                const optimizerConfig = parsedJson.trainingConfig.optimizer_config;
                let optimizer;
                
                console.log('Optimizer config:', optimizerConfig);
                
                if (optimizerConfig.class_name === 'Adam') {
                    // learning_rateを数値として取得
                    let lr = 0.001; // デフォルト値
                    if (optimizerConfig.config && typeof optimizerConfig.config.learning_rate !== 'undefined') {
                        lr = Number(optimizerConfig.config.learning_rate);
                        console.log('Learning rate from config:', lr, 'type:', typeof lr);
                    }
                    
                    // 数値であることを確認
                    if (isNaN(lr) || !isFinite(lr)) {
                        console.warn('Invalid learning rate, using default 0.001');
                        lr = 0.001;
                    }
                    
                    optimizer = tf.train.adam(lr);
                } else {
                    optimizer = tf.train.adam(0.001); // デフォルト
                }
                
                const loss = parsedJson.trainingConfig.loss || 'categoricalCrossentropy';
                model.compile({ 
                    optimizer: optimizer, 
                    loss: loss, 
                    metrics: ['accuracy'] 
                });
                console.log('Model compiled successfully');
            } else {
                // trainingConfigがない場合はデフォルト設定でコンパイル
                model.compile({ 
                    optimizer: tf.train.adam(0.001), 
                    loss: 'categoricalCrossentropy', 
                    metrics: ['accuracy'] 
                });
                console.log('Model compiled with defaults');
            }
            
            model.summary();
            
            // クラス名と学習履歴を復元
            if (parsedJson.userDefinedMetadata) {
                if (parsedJson.userDefinedMetadata.classNames) {
                    classNames = parsedJson.userDefinedMetadata.classNames;
                    console.log('Restored class names:', classNames);
                }
                
                // 学習履歴を復元
                if (parsedJson.userDefinedMetadata.trainingHistory) {
                    trainingHistory.epochs = parsedJson.userDefinedMetadata.trainingHistory.epochs || [];
                    trainingHistory.losses = parsedJson.userDefinedMetadata.trainingHistory.losses || [];
                    console.log(`Restored training history: ${trainingHistory.epochs.length} epochs`);
                    
                    // チャートを復元
                    restoreChart();
                }
                
                updateStatus(`Model restored with ${trainingHistory.epochs.length} epochs of training. Ready for inference or further training.`, false);
                isDirty = false;
            } else {
                classNames = [];
                trainingHistory = { epochs: [], losses: [] };
                updateStatus('Model layers and compiler restored. Please select a dataset to define classes.', false);
                console.warn('No metadata found in model.');
                isDirty = false;
            }

        } catch (error) {
            alert(`Failed to load model: ${error.message}`);
            console.error('Full error:', error);
            console.error('Stack trace:', error.stack);
            updateStatus('Failed to load model.', false);
        } finally {
            event.target.value = '';
        }
    }
});