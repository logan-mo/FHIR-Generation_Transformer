import torch, torchtext
from torch import nn
# Define the tokenizer
tokenizer = torchtext.data.get_tokenizer('basic_english')

# Tokenize the data
tokenized_data = []
with open('data/chunk_1.csv', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        tokenized_data.append(tokenizer(line[:]))

# Build the vocabulary
vocab = torchtext.vocab.build_vocab_from_iterator(lines)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Convert tokens to indices
sequences = []
for token_sequence in tokenized_data:
    sequences.append([vocab[token] for token in token_sequence])

# Define the model
embedding_dim = 64
hidden_dim = 128
vocab_size = 105506

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output


# Initialize the model and move it to the device


model = RNNModel(vocab_size, embedding_dim, hidden_dim)

# Define your seed sequence
seed_sequence = ['[resourcetype]careplan']
#CarePlan,"[resourceType] CarePlan [id] 5026185d-3747-e969-6564-c4aec1b8d06e [meta][profile][0] http://hl7.org/fhir/us/core/StructureDefinition/us-core-careplan [text][status] generated [text][div] <div xmlns=""http://www.w3.org/1999/xhtml"">Care Plan for Infectious disease care plan (record artifact).<br/>Activities: <ul><li>Infectious disease care plan (record artifact)</li><li>Infectious disease care plan (record artifact)</li><li>Infectious disease care plan (record artifact)</li></ul><br/>Care plan is meant to treat Suspected COVID-19.</div> [status] completed [intent] order [category][0][coding][0][system] http://hl7.org/fhir/us/core/CodeSystem/careplan-category [category][0][coding][0][code] assess-plan [category][1][coding][0][system] http://snomed.info/sct [category][1][coding][0][code] 736376001 [category][1][coding][0][display] Infectious disease care plan (record artifact) [category][1][text] Infectious disease care plan (record artifact) [subject][reference] Patient/2b7d4554-d6e4-9f48-2ab0-0ddf088fe19d [encounter][reference] Encounter/723aa866-47af-c57d-fe71-cebfba9b15cf [period][start] 2020-02-28T22:59:13-05:00 [period][end] 2020-03-14T02:15:17-04:00 [careTeam][0][reference] CareTeam/49500403-2881-84d8-cb65-015bfea4c65d [addresses][0][reference] Condition/c925509e-0334-6544-6af4-e6ffa96144b3 [activity][0][detail][code][coding][0][system] http://snomed.info/sct [activity][0][detail][code][coding][0][code] 444908001 [activity][0][detail][code][coding][0][display] Isolation nursing in negative pressure isolation environment (regime/therapy) [activity][0][detail][code][text] Isolation nursing in negative pressure isolation environment (regime/therapy) [activity][0][detail][status] completed [activity][0][detail][location][display] NEWTON-WELLESLEY HOSPITAL [activity][1][detail][code][coding][0][system] http://snomed.info/sct [activity][1][detail][code][coding][0][code] 409524006 [activity][1][detail][code][coding][0][display] Airborne precautions (procedure) [activity][1][detail][code][text] Airborne precautions (procedure) [activity][1][detail][status] completed [activity][1][detail][location][display] NEWTON-WELLESLEY HOSPITAL [activity][2][detail][code][coding][0][system] http://snomed.info/sct [activity][2][detail][code][coding][0][code] 409526008 [activity][2][detail][code][coding][0][display] Personal protective equipment (physical object) [activity][2][detail][code][text] Personal protective equipment (physical object) [activity][2][detail][status] completed [activity][2][detail][location][display] NEWTON-WELLESLEY HOSPITAL"
# Convert the seed sequence to a tensor of indices
seed_tensor = torch.tensor([vocab[token] for token in seed_sequence]).unsqueeze(0).to(device)

# Set the model to evaluation mode
model.eval()

# Initialize the generated sequence with the seed sequence
generated_sequence = seed_sequence

# Generate a sequence of length 100
for _ in range(100):
    with torch.no_grad():
        # Get the model's prediction for the next token
        output = model(seed_tensor)

        # Get the index of the predicted token
        predicted_index = output.argmax(dim=-1)[0, -1].item()

        # Convert the index to a token
        predicted_token = vocab.get_itos()[predicted_index]

        # Add the predicted token to the generated sequence
        generated_sequence.append(predicted_token)

        # Add the predicted index to the seed tensor
        seed_tensor = torch.cat([seed_tensor, torch.tensor([[predicted_index]]).to(device)], dim=1)

# Print the generated sequence
print(' '.join(generated_sequence))
