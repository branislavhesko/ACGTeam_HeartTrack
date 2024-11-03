---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: /images/20241102_112341.jpg
# some information about your slides (markdown enabled)
title: HeartTrack
info: |
  ## Hack Jak Brno
  ACGTeam solutions s.r.o.

  Hesko, Kadlec, Kohútek, Šustr
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# take snapshot for each slide in the overview
overviewSnapshots: true
---

# HeartTrack

ACGTeam solutions s.r.o.

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space to dive into ACGTeam work <carbon:arrow-right class="inline"/>
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/branislavhesko/ACGTeam_HeartTrack" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
transition: fade-out
---

# ACGTeam

  - Branislav Hesko
    - Učím elektrónové mikroskopy vidieť a myslieť
  - Vojtěch Kadlec
    - Učím modely aby ľudia videli a neoslepli
  - Michal Kohútek
    - Nitko tú korporátnu JAVu musí robiť
  - Michal Šustr
    - Teoria hier je moja parketa a porazím ťa v black jacku


  **Prišli sme sem bez prípravy a vytvorili sme funkčnú HeartTrack aplikáciu**



---
transition: fade-in
---

# Čo sa podarilo?

 - **Funkčný** prototyp aplikácie pre meranie srdečného tepu pomocou kamery a mikrofónu.
 - Ukladanie a online vyhodnotenie dát pre účely telemedicíny
 - Meranie srdečného tepu s presnosťou na 2 tepy v porovnaní s Apple Watch
 - Aplikácia spoľahlivo funguje na 6 rokov starom mobile
 - https://github.com/branislavhesko/ACGTeam_HeartTrack
  
---
layout: default
---

# Ako to funguje?
- Mobilná aplikácia pravidelne odosiela namerané video so zvukom na server
- Na serveri ich vyhodnocujú 3 AI modely: kvalita, detekcia tepu a detekcia tepu z mikofónu.
- Výsledky sú zobrazované v aplikácii a vo webovej verzii i podľa pacienta


---
layout: cover
---

# Čas na demo


---
layout: default
---

# Čo ďalej?
- 📝 Spracovanie uložených historických dát pre konkrétneho pacienta
- 🪄 Vylepšenie AI modelov pre detekciu tepu
- 📱 Integrácia modelov on device
- 🎨 Vylepšenie UI a port na iPhone

