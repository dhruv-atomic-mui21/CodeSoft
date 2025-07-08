import torch
from PIL import Image
from torchvision import transforms

def load_model_for_inference(checkpoint_path, vocab_size, embed_size=512, hidden_size=1024):
    from src.model import ImageCaptioningModel
    model = ImageCaptioningModel(vocab_size=vocab_size, embed_size=embed_size, hidden_size=hidden_size)
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def process_image_for_inference(image_path, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        return image.to(device)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def generate_caption_beam_search(image_tensor, model, sp, beam_width=3, max_len=30, device=None):
    if model is None:
        print("Model is not loaded. Cannot generate caption.")
        return "Error: Model not loaded."
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = model.encoder(image_tensor)
        sequences = [[list(), 0.0, (torch.zeros(1, 1, model.decoder.lstm.hidden_size).to(device),
                                  torch.zeros(1, 1, model.decoder.lstm.hidden_size).to(device))]]
        start_token = sp.bos_id()
        end_token = sp.eos_id()
        for _ in range(max_len):
            all_candidates = []
            for seq, score, (h, c) in sequences:
                if len(seq) > 0 and seq[-1] == end_token:
                    all_candidates.append((seq, score, (h, c)))
                    continue
                if len(seq) == 0:
                    inputs = model.decoder.embed(torch.tensor([start_token]).to(device)).unsqueeze(1)
                else:
                    inputs = model.decoder.embed(torch.tensor([seq[-1]]).to(device)).unsqueeze(1)
                attention_weighted_encoding, _ = model.decoder.attention(features, h.squeeze(0))
                lstm_input = torch.cat((inputs.squeeze(1), attention_weighted_encoding), dim=1).unsqueeze(1)
                output, (h, c) = model.decoder.lstm(lstm_input, (h, c))
                output = model.decoder.linear(output.squeeze(1))
                log_probs = torch.log_softmax(output, dim=1)
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                for i in range(beam_width):
                    candidate = seq + [top_indices[0][i].item()]
                    candidate_score = score - top_log_probs[0][i].item()
                    all_candidates.append((candidate, candidate_score, (h, c)))
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:beam_width]
            if all(seq[-1] == end_token for seq, _, _ in sequences):
                break
        best_seq = sequences[0][0]
        tokens = [sp.id_to_piece(id) for id in best_seq if id not in (sp.pad_id(), sp.bos_id(), sp.eos_id())]
        return ' '.join(tokens)

def generate_caption_greedy_search(image_tensor, model, sp, max_len=30, device=None):
    if model is None:
        print("Model is not loaded. Cannot generate caption.")
        return "Error: Model not loaded."
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = model.encoder(image_tensor)
        caption = []
        token = sp.bos_id()
        end_token = sp.eos_id()
        avg_features = features.mean(dim=1)
        hidden_state = model.decoder.init_h(avg_features).unsqueeze(0)
        cell_state = model.decoder.init_c(avg_features).unsqueeze(0)
        for _ in range(max_len):
            inputs = model.decoder.embed(torch.tensor([token]).to(device)).unsqueeze(1)
            attention_weighted_encoding, _ = model.decoder.attention(features, hidden_state.squeeze(0))
            lstm_input = torch.cat((inputs.squeeze(1), attention_weighted_encoding), dim=1).unsqueeze(1)
            output, (hidden_state, cell_state) = model.decoder.lstm(lstm_input, (hidden_state, cell_state))
            output = model.decoder.linear(output.squeeze(1))
            _, predicted_token = torch.max(output, dim=1)
            token = predicted_token.item()
            if token == end_token:
                break
            if token != sp.pad_id() and token != sp.bos_id():
                caption.append(sp.id_to_piece(token))
        return ' '.join(caption)
