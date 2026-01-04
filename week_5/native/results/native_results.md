| Model (Native, no RAG)       | Behavior when asked about Angeloktistisi Church                                      | Quality (1–10) | Special Obsession                  |
|------------------------------|--------------------------------------------------------------------------------------|----------------|------------------------------------|
| SmolLM-135M                  | Total fiction: built in 1976, labyrinth, 3 gates ("Forgery", "Shattered", "Vast Chamber"), treasures | 0              | Elevators & sewer systems          |
| SmolLM-360M                  | Wild hallucinations: church in Kyiv (1874), Smyrna on Kerameikos, tunnels, elevators, sewers     | 1              | Byzantine elevators & sewage mains |
| Qwen3-0.6b                   | Doesn’t know the church, invents plausible-sounding details (Turkey, 12th century, fountain, etc.) | 4              | Mild invention                     |
| Granite-4.0-h-350m           | Honestly says “I don’t know”, refuses to make up facts                               | 8              | Zero hallucinations                |

**Key takeaway (native mode):**
- SmolLM (135M & 360M) → completely unusable without RAG, 100 % hallucinations
- Qwen3-0.6b → tries to answer, but still fabricates
- Granite-350m → the only tiny model that doesn’t lie when it doesn’t know

Save as `native_test_results.md`
