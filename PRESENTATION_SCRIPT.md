# Presentation Script

This is a casual 5-6 minute repo walkthrough script. I plan to speak from the README, not from slides.

## 1. Research Question

Today I am going to walk through my Tibetan-to-Mandarin speech-to-unit translation prototype.

The main question is pretty simple:

> Can Tibetan speech be translated toward Mandarin speech-like units without first building a full Tibetan ASR system?

So instead of doing:

```text
Tibetan speech -> Tibetan text -> Mandarin text -> Mandarin speech
```

I try a smaller alternative:

```text
Tibetan speech features -> Mandarin acoustic unit sequence
```

This is inspired by Gong, Xu, and Zhao's 2025 paper on Tibetan-Chinese speech-to-speech translation with discrete units. My version is much smaller and more like a course-scale prototype, but it tests the same basic idea: maybe we can avoid making Tibetan text the required middle step.

## 2. Motivation

The motivation I use is a medical access scenario.

Imagine an elderly Tibetan speaker from a remote pastoral area going to a large hospital in Chengdu. The hospital staff mostly speak Mandarin. The patient may need to explain pain, symptoms, medication history, or follow-up questions in Tibetan, but the hospital system works through Mandarin.

For me, this makes the project more than a benchmark problem. It is also an access problem. A translation system in this setting can be helpful only if it preserves meaning. If it produces fluent Mandarin that says the wrong thing, then it may actually make the situation more dangerous.

So I do not want to present this as a medical interpreter. It is not deployable. I see it more as a small experiment asking: can this speech-to-unit idea reduce one bottleneck, and where does it still fail?

## 3. Why Tibetan-to-Chinese Speech Translation Is Hard

The usual pipeline for speech translation is:

```text
speech -> ASR text -> machine translation text -> target speech
```

This is clean in theory, but it becomes fragile here.

First, it needs reliable Tibetan ASR. That is already hard because Tibetan speech has dialectal variation, limited transcribed data, and a gap between spoken forms and standardized written forms.

Second, the writing-system part is not as simple as "audio becomes a line of characters." This is where I would point to the first SVG in the README.

The Latin example shows the intuition many of us have from letter-based writing: a mostly linear sequence like c-a-t. The Korean example shows that Hangul is also compositional, but Korean has much stronger normalization tools, segmentation conventions, and data support. The Tibetan example shows a written syllable cluster, where base letters, signs, suffixes, stacks, and syllable delimiters can all matter.

The point is not that Tibetan cannot be represented in Unicode. It can. The problem is that speech-to-written alignment, normalization, and segmentation are much less supported in this low-resource setting. So if the whole system depends on first producing clean Tibetan text, that first step becomes a major bottleneck.

There is also a second problem: even after ASR and translation, a full system would still need Mandarin speech generation. That means more data, more modeling, and usually a vocoder. For this project, I wanted something smaller and more honest.

## 4. Core Idea: How I Understand S2UT

The core idea of S2UT, or speech-to-unit translation, is:

> Do not predict text as the target. Predict speech-like units.

In my pipeline, the source side is Tibetan speech. I convert the Tibetan audio into 80-dimensional log-Mel features. I think of these as compact speech features: they describe the sound over time without using raw waveform samples directly.

The target side starts from Mandarin text in the dataset. But because I need a speech-like target, I first use Edge-TTS to synthesize Mandarin audio from that Mandarin text.

Then I pass the Mandarin audio through HuBERT. HuBERT is a pretrained speech model, and I use its layer-6 features. These are intermediate speech representations: not raw audio, not text, but learned acoustic or phonetic features.

After that, I use K-means clustering. For K=100, the idea is that Mandarin speech frames are grouped into 100 acoustic categories. So Mandarin speech becomes a sequence of unit IDs, something like:

```text
16 24 55 77 45
```

These numbers are not words. They are learned speech categories.

During training, the model sees paired examples:

```text
input: Tibetan speech features
target: Mandarin unit sequence
```

It is not trained with explicit word-level or frame-level alignment. It is trained more like a translation model. The Transformer encoder reads the Tibetan speech features, and the decoder learns to predict the Mandarin unit sequence. Any alignment between the Tibetan audio and the Mandarin units is learned implicitly through attention.

I also add a small 3-gram unit language model during decoding. This language model does not understand Tibetan or Mandarin meaning. It only says which unit sequences look locally plausible. So it can improve unit fluency, but it cannot guarantee semantic correctness.

## 5. Results and Analysis

For the experiment, I use the TCST Tibetan-Chinese speech translation dataset. After preprocessing, I have about 5,800 training utterances, 725 dev utterances, and 726 test utterances.

I tested different unit inventories: K=100, 200, 500, and 1000.

The best setting is K=100. With greedy decoding, K=100 gets 12.10 Unit-BLEU. With the 3-gram unit language model, and LM weight 0.6, it reaches 19.50 Unit-BLEU.

So numerically, the LM helps a lot inside this experiment. It improves the score by 7.40 Unit-BLEU.

But I do not want to overstate this number. Unit-BLEU is not the same as semantic translation quality. It measures overlap between unit sequences, not whether the Mandarin sentence means the same thing as the Tibetan speech.

That is why I added retrieval diagnostics. The idea is: take the predicted Mandarin unit sequence, find the closest unit sequence in the training set, and look at the retrieved Mandarin text. If the retrieved sentence is semantically unrelated, then the model should not be treated as successful just because the unit score improved.

One bad case is:

```text
Reference Mandarin:
不断加大高水平对外开放力度，

Retrieved Mandarin:
今天有人买牙膏吗？
```

These are completely different meanings.

Another pattern I saw is that different test examples can retrieve the same unrelated Mandarin phrase:

```text
香港特区政府发言人说，
```

This suggests that the model may collapse toward a locally plausible unit pattern instead of really preserving the source meaning.

So the main result is mixed. The system is trainable, and the K=100 plus language-model setting is clearly better by Unit-BLEU. But the qualitative cases show that this number is not enough for real use.

In a medical access scenario, fluent but wrong Mandarin is still a failure.

## 6. Conclusion

My takeaway is that speech-to-unit translation is a useful idea for low-resource settings because it gives another option besides ASR-first translation.

But this project also shows the risk of stopping at a metric. A model can improve on Unit-BLEU and still fail semantically.

So I would describe the contribution as two things:

First, I built a small reproducible prototype that maps Tibetan speech features to Mandarin acoustic units.

Second, I showed why honest qualitative diagnostics matter, especially when the imagined use case is access to medical or public services.

The system is not ready for deployment. But it makes the bottleneck visible, and it gives a concrete starting point for thinking about how low-resource speech translation could be evaluated more carefully.
