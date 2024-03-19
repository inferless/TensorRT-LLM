# Triton TensorRT-LLM
TensorRT-LLM can be used through the docker container. There are multiple ways you can do it:
TensorRT-LLM can be accessed via Docker container, offering two approaches:
1. Clone the [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) GitHub repository and build the container.
2. Alternatively, clone the [Tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend) GitHub repository, which includes TensorRT-LLM. This option is preferred as it avoids potential TensorRT version mismatches during model inference.

## FOLLOW THE FOLLOWING STEP TO BUILD THE CONTAINER:

### STEP 1. Clone [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend.git)
```
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git -b v0.8.0
cd tensorrtllm_backend
git lfs install
git submodule update --init --recursive
```

### STEP 2. Build the Triton TRT-LLM backend container

```
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f dockerfile/Dockerfile.trt_llm_backend .
```
Note: Building the container will take some time < 2 hours.

### STEP 3: Model Preperation
Before lunching the container, create a directory `model_input` and download the huggingface model into it. Then create `model_output` directory, where we will store the model engine. Make sure that you are on your home directory.

```bash
mkdir model_input
cd model_input
git lfs install
# For gated model use this -> git clone https://<user_name>:<hf_token>@huggingface.co/meta-llama/Llama-2-7b-hf
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
# Move to the home directory
cd ..
mkdir model_output
cd model_output
mkdir model_checkpoint
cd ..
# Again create another directory for model_engine
mkdir model_engine
```
### STEP 4. Launch the docker container
```
docker run --gpus all --network host -it --shm-size=1g -v $(pwd)/tensorrtllm_backend/all_models:/all_models -v $(pwd)/tensorrtllm_backend/scripts:/opt/scripts -v ${PWD}/model_input:/model_input -v ${PWD}/model_output:/model_output triton_trt_llm bash
```

### STEP 5. Build engines
Once you are inside the docker container, you can now move to `tensorrt_llm/examples` to convert the model into tensorrt-llm checkpoint format and build the engine.
```
cd tensorrt_llm/examples/llama
# Convert weights from HF Tranformers to tensorrt-llm checkpoint format
python convert_checkpoint.py --model_dir /model_input/Llama-2-7b-hf/ \
                             --output_dir /model_input/model_checkpoint/ \
                             --dtype float16

# Build TensorRT engines
trtllm-build --checkpoint_dir /model_input/model_checkpoint/ \
            --output_dir /model_output/model_engine/ \
            --gemm_plugin float16 \
            --max_input_len 4000 \
            --max_output_len 4000
```

### STEP 6. Prepare the Model for inference
Copy all the model assets from `model_output` to `tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1/`:

```bash
cp -r /model_output/model_engine/* /all_models/inflight_batcher_llm/tensorrt_llm/1/
```

### 6. Update the model configuration files
Tensorrtllm_backend provide a script to modify the configuration files. Run the following commands in the `tensorrtllm_backend` directory:

1. Preprocessing: `/tensorrtllm_backend/all_models/inflight_batcher_llm/preprocessing/config.pbtxt`

#### Run the following command to update preprocessing config.pbtxt
```
python3 tools/fill_template.py -i all_models/inflight_batcher_llm/preprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:meta-llama/Llama-2-7b-hf,tokenizer_type:llama,preprocessing_instance_count:1"
```
2. Tensorrt_llm: `/tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt`

#### Run the following command to update tensorrt_llm config.pbtxt
```
python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt "triton_max_batch_size:4,decoupled_mode:False,engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1/,batching_strategy:V1,max_queue_delay_microseconds:100"
```
3. Postprocessing: `/tensorrtllm_backend/all_models/inflight_batcher_llm/postprocessing/config.pbtxt`

#### Run the following command to update postprocessing config.pbtxt
```
python3 tools/fill_template.py -i all_models/inflight_batcher_llm/postprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:meta-llama/Llama-2-7b-hf,tokenizer_type:llama,postprocessing_instance_count:1"
```
4. Ensemble: `triton_model_repo/ensemble/config.pbtxt`

#### Run the following command to update ensemble config.pbtxt
```
python3 tools/fill_template.py -i all_models/inflight_batcher_llm/ensemble/config.pbtxt "triton_max_batch_size:4"
```

### 7. Launch Triton server

Time to launch the Triton server!

```
python3 /opt/scripts/launch_triton_server.py --model_repo /all_models/inflight_batcher_llm --world_size 1
```
When successfully deployed, the server produces logs similar to the following ones.

```
I0919 14:52:10.475738 293 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0919 14:52:10.475968 293 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0919 14:52:10.517138 293 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```
### 8. Test the server with the Triton generate endpoint:

```
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'
```
