import torch
from torch import nn
def train(tokenizer, model, device, lr, EPOCH, loader):

    # setup optimizer and scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 150)

    for epoch in range(EPOCH):
        step = 0
        print('Epoch:', epoch + 1, 'Training...')

        for step, (text, label) in enumerate(loader): 

            list_text = list(text)
            text_tokens = tokenizer(list_text, return_tensors="pt", padding=True)
            input_ids = text_tokens['input_ids'].to(device)
            attention_mask = text_tokens['attention_mask'].to(device)

            logits_user = model(input_ids, attention_mask)
            
            label = label.to(device)
            loss = criterion(logits_user, label)
            print("loss:", loss, "step:", step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    torch.save(model, "classify_roberta.bin")