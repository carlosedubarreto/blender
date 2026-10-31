path = 'source/blender/draw/engines/workbench/workbench_engine.cc'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """    /* Pass 1 records all compute traversals; pass 2 issues all indirect draws. Interleaving
     * compute and graphics per object forced Vulkan to end/restart the rendering scope and
     * insert compute->graphics barriers once per object. */
    struct TesseraPendingDraw {
      TesseraObjectState *state;
      bool allow_dna_write;
    };
    Vector<TesseraPendingDraw> pending_draws;"""

replacement1 = """    auto t_p1_start = std::chrono::steady_clock::now();

    /* Pass 1 records all compute traversals; pass 2 issues all indirect draws. Interleaving
     * compute and graphics per object forced Vulkan to end/restart the rendering scope and
     * insert compute->graphics barriers once per object. */
    struct TesseraPendingDraw {
      TesseraObjectState *state;
      bool allow_dna_write;
    };
    Vector<TesseraPendingDraw> pending_draws;"""

assert target1 in content, "target1 not found"
content = content.replace(target1, replacement1, 1)

target2 = """    for (const TesseraPendingDraw &pending : pending_draws) {"""

replacement2 = """    auto t_p1_end = std::chrono::steady_clock::now();

    for (const TesseraPendingDraw &pending : pending_draws) {"""

assert target2 in content, "target2 not found"
content = content.replace(target2, replacement2, 1)

target3 = """    /* Aggregated drawn-instance counts: one write per emitter object so the
     * stat always reflects the whole scatter field across all archetypes. */"""

replacement3 = """    auto t_p2_end = std::chrono::steady_clock::now();
    static int s_timing_frame = 0;
    if (++s_timing_frame % 10 == 0) {
      float ms_p1 = std::chrono::duration<float, std::milli>(t_p1_end - t_p1_start).count();
      float ms_p2 = std::chrono::duration<float, std::milli>(t_p2_end - t_p1_end).count();
      printf("[Tessera Engine Timing] Pass1 (Traverse/Cull): %6.2f ms | Pass2 (Draw Issue): %6.2f ms | Pending Draws: %d\\n",
             ms_p1, ms_p2, int(pending_draws.size()));
      fflush(stdout);
    }

    /* Aggregated drawn-instance counts: one write per emitter object so the
     * stat always reflects the whole scatter field across all archetypes. */"""

assert target3 in content, "target3 not found"
content = content.replace(target3, replacement3, 1)

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print("Instrumented workbench_engine.cc successfully")

