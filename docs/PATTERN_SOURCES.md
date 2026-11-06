# Pattern sources

`triadq/patterns.json` holds statistics measured on the hand-made meshes below: valence shares, corner pattern counts, medians per shape class. It holds no vertices, faces or other geometry from them. Rebuild it with `scripts/build_patterns.py` in the CurioMesh source repository.

| Source | Use | License | Attribution | Meshes used |
|---|---|---|---|---|
| [Blender Studio Human Base Meshes v1.4.1](https://www.blender.org/download/demo/) | character | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Blender Studio, Human Base Meshes | 82 |
| MakeHuman base mesh (`base.obj`, body only) | character | CC0 1.0 (assets) | MakeHuman Community, base mesh | 1 |
| [Keenan Crane's 3D Model Repository](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/): Spot, Bob, Blub, Jerry the Ogre (control meshes only) | character | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Keenan Crane's 3D Model Repository (Spot: Crane, Pinkall and Schroeder 2013) | 4 |
| [Blender Studio Sprite (Sprite Fright)](https://studio.blender.org/characters/sprite/v1/) | character | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Sprite Rig (CC-BY) Blender Foundation, studio.blender.org | 14 |
| [Blender Studio Rain](https://studio.blender.org/characters/rain/v3/) | character | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Rain Rig (CC) Blender Foundation \| studio.blender.org | 17 |
| [Blender Studio Pack Bot (Charge)](https://studio.blender.org/characters/pack-bot/v1/) | hard surface | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Pack Bot Rig (CC-BY) Blender Foundation, studio.blender.org | 40 |
| [Blender Studio Security Bot (Charge)](https://studio.blender.org/characters/security-bot/v1/), parts over 95% quads | hard surface | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Security Bot Rig (CC-BY) Blender Foundation, studio.blender.org | 89 |

The numbers were derived from these works, and changed from them by reducing them to statistics. The works are provided as-is under their licenses, without warranties; see each license for details.

## What this set can't tell you

Most of it is characters, and the hard-surface part comes from two robots. Nothing here shows how an artist builds a rock, a ship or a large machine, so the score says how far a mesh is from this kind of work, not what is right for those. Raw base meshes were measured, without their modifiers: a part that gets its roundness from a Bevel or Subdivision modifier looks plainer here than it does in the final model.

The score only judges tidiness. A neat grid that misses the surface would score well, so it is always read together with the camera check.
