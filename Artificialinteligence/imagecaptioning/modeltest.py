import sentencepiece as spm

sp = spm.SentencePieceProcessor()
ok = sp.load('Artificialinteligence/imagecaptioning/bpe.model')
print("✅ Loaded successfully" if ok else "❌ Load failed")
