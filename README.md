# MoGPT: An Empirical Study of Multitask Learning to Improve Open Domain Dialogue Systems

## Abstract

Autoregressive models used to generate responses in open-domain dialogue systems often struggle to take long-term context into account and to maintain consistency over a dialogue.
Previous research in open-domain dialogue generation has shown that the use of *auxiliary tasks* can introduce inductive biases that encourage the model to improve these qualities. However, most previous research has focused on encoder-only or encoder/decoder models, while the use of auxiliary tasks in *decoder-only* autoregressive models is under-explored.
This paper describes an investigation where four different auxiliary tasks are added to small and medium-sized GPT-2 models fine-tuned on the PersonaChat and DailyDialog datasets.
The results show that the introduction of the new auxiliary tasks leads to small but consistent improvement in evaluations of the investigated models.

## Usage

...

## Fine-Tuned Model Weights
The fine-tuned weights for the models on both PersonaChat and DailyDialog datasets can be found on the Hugging Face Model Hub. Follow the links provided below to access them:

- **PersonaChat Fine-Tuned Models**: 
  - Small: [https://huggingface.co/m3hrdadfi/MoGPT-experiments-dd-small](https://huggingface.co/m3hrdadfi/MoGPT-experiments-dd-small)
  - Medium: [https://huggingface.co/m3hrdadfi/MoGPT-experiments-dd-medium](https://huggingface.co/m3hrdadfi/MoGPT-experiments-dd-medium)
- **DailyDialog Fine-Tuned Models**: 
  - Small: [https://huggingface.co/m3hrdadfi/MoGPT-experiments-pc-small](https://huggingface.co/m3hrdadfi/MoGPT-experiments-pc-small)
  - Medium: [https://huggingface.co/m3hrdadfi/MoGPT-experiments-pc-medium](https://huggingface.co/m3hrdadfi/MoGPT-experiments-pc-medium)

*These models can then be used for generating responses in your dialogue systems. Please refer to the Hugging Face documentation and the Transformers library for more information on how to use these models.*

## Reference
```bibtex
@inproceedings{MoGPT,
    title = "An Empirical Study of Multitask Learning to Improve Open Domain Dialogue Systems",
    author = "Farahani, Mehrdad  and Johansson, Richard",
    booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may,
    year = "2023",
    address = "T{\'o}rshavn, Faroe Islands",
    publisher = "University of Tartu Library",
    url = "https://aclanthology.org/2023.nodalida-1.36",
    pages = "347--357",
    abstract = "Autoregressive models used to generate responses in open-domain dialogue systems often struggle to take long-term context into account and to maintain consistency over a dialogue. Previous research in open-domain dialogue generation has shown that the use of \textit{auxiliary tasks} can introduce inductive biases that encourage the model to improve these qualities. However, most previous research has focused on encoder-only or encoder/decoder models, while the use of auxiliary tasks in \textit{encoder-only} autoregressive models is under-explored. This paper describes an investigation where four different auxiliary tasks are added to small and medium-sized GPT-2 models fine-tuned on the PersonaChat and DailyDialog datasets. The results show that the introduction of the new auxiliary tasks leads to small but consistent improvement in evaluations of the investigated models.",
}
```
## Contributing
We welcome contributions. Please submit a Pull Request and we will review your contribution.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Feel free to create an issue for any questions or problems. We will try our best to help.
