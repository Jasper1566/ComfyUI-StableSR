# comfyui-stablsr插件

## 使用说明
从huggingface仓库 https://huggingface.co/Iceclear/StableSR/ 或百度盘 https://pan.baidu.com/s/10OAff20AtNhNFfJYgN1F6Q?pwd=ox5o 
下载webui_786v_139.ckpt和stablesr_768v_000139.ckpt两个模型文件

下载后StableSR webui_786v_139.ckpt 放到 Comyfui/models/stablesr/
StableSR stablesr_768v_000139.ckpt 放到 Comyfui/models/checkpoints/
文件夹不存在的话自行创建即可

如果你的comfyui模型目录设置了共享webui的模型目录，则放到webui对应目录下



## usage
Download webui_786v_139.ckpt and stablesr_768v_000139.ckpt, from https://huggingface.co/Iceclear/StableSR/ or from Baidu Pan at https://pan.baidu.com/s/10OAff20AtNhNFfJYgN1F6Q?pwd=ox5o

then put StableSR webui_786v_139.ckpt into  Comyfui/models/stablesr/ 
and StableSR stablesr_768v_000139.ckpt into Comyfui/models/checkpoints/ 
If the folders do not exist, create them yourself.

If you have set the shared webui model directory for your comfyui, place the files in the corresponding webui directory.

There is a setup json in /examples/ to load the workflow into Comfyui
