# TODO:

- [x] Setup a readme with a quick start guide.
- [x] setup continuous integration and pipy deployment.
- [x] Improve document conversion (page per page text and video interleaving + whole page images)
- [x] Cleanup structured output with tool usage
- [x] Implement streaming callbacks
- [x] Improve folder/archive conversion: add ascii folder tree
- [x] Reorganise media files used for testing into a single media folder
- [x] RAG (still missing ingestion code for arbitary digital objects: folders, pdf, images, urls, etc...)
- [x] Improve logging with arbol, with option to turn off.
- [x] Use specialised libraries for document type identification
- [ ] Use the faster pybase64 for base64 encoding/decoding.
- [ ] Automatic feature support discovery for models (which models support images as input, reasoning, etc...)
- [ ] Deal with message sizes in tokens sent to models
- [ ] Improve vendor api robustness features such as retry call when server errors, etc...
- [ ] Improve and uniformize exception handling
- [ ] Add support for adding nD images to messages.
- [ ] Implement 'brainstorming' mode for text generation, possibly with API fusion.
- [ ] Response format option for agent.
