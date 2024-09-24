# datago
A golang-based data loader which can be used from Python. Compatible with a soon-to-be open sourced VectorDB-enabled data stack, which exposes HTTP requests. 

Datago will handle, outside of the Python GIL 
- per sample IO from object storage
- deserialization
- some optional vision processing (aligning different image payloads)
- serialization

Samples are then exposed in the Python scope and ready for consumption, typically using PIL and Numpy base types.
Speed will be network dependent, but GB/s is relatively easily possible

<img width="922" alt="Screenshot 2024-09-24 at 9 39 44 PM" src="https://github.com/user-attachments/assets/b58002ce-f961-438b-af72-9e1338527365">
