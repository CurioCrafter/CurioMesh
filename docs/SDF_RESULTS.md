# SDF thickness tests

A short record of what was tried and how it came out, so nobody has to try it again from memory.

## The idea

Protect thin parts by measuring thickness with a signed distance field, and use that as vertex weights in Decimate, so thin areas keep more faces.

## First test: thickness through Decimate weights

It scored worse than plain Decimate: 0.41% lost detail against 0.29%. Pictures showed why: thickness starved a tank, flattened a hose and made a stray spike. It was not kept.

## Later test

Lost detail improved by 2.7x on the compressor and 1.6x on the chest. Together with the camera check it was 3.1x and 3.0x. This is a partial test, because the inward rays used to measure thickness often missed. Don't read it as proof that SDF thickness works.

## Where it stands

Thickness doesn't steer Auto Segments. Density in tight areas comes from where the camera saw the view go wrong, with no thickness measure.
