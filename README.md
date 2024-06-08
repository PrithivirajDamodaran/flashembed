<center>
<h1> What is FlashEmbed? </h1>
</center>

Lightweight & Fast Python library to add low-footprint (all-MiniLM-* equivalent) multilingual retrievers to your RAG and Search & Retrieval pipelines. No heavy torch or transformer dependencies like it's Sister library  [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank). FlashEmbed uses miniMiracle* series of models. Ofcourse we will be adding more retrievers in future.

<h2> 📖 License & Terms </h2>  

The library is licensed under Apache 2.0 but the weights are licensed differently see below for details. Note: The below license & terms apply ONLY for miniMiracle series models. Use responsibly.

<center>
<img src="./images/terms.png" width=80%>
</center>


<h2> 🚀 Installation </h2> 

```python 
pip install flashembed
```
<h2> Supported Models </h2>

- [prithivida/miniMiracle_hi_v1](https://huggingface.co/prithivida/miniMiracle_hi_v1)
- [prithivida/miniMiracle_te_v1](https://huggingface.co/prithivida/miniMiracle_te_v1)
- [prithivida/miniMiracle_zh_v1](https://huggingface.co/prithivida/miniMiracle_zh_v1)


<h2> 📖 Usage </h2>  

```python
from flashembed import Embedder
from typing import List

passages = [
    'एक आदमी खाना खा रहा है।',
    'लोग ब्रेड का एक टुकड़ा खा रहे हैं।',
    'लड़की एक बच्चे को उठाए हुए है।',
    'एक आदमी घोड़े पर सवार है।',
    'एक महिला वायलिन बजा रही है।',
    'दो आदमी जंगल में गाड़ी धकेल रहे हैं।',
    'एक आदमी एक सफेद घोड़े पर एक बंद मैदान में सवारी कर रहा है।',
    'एक बंदर ड्रम बजा रहा है।',
    'एक चीता अपने शिकार के पीछे दौड़ रहा है।',
    'एक बड़ा डिनर है।'
]
    

# Onetime Init and Load model
embedder = Embedder('prithivida/miniMiracle_hi_v1')

embeddings = embedder.encode(passages) 

