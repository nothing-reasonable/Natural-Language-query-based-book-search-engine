# Book categories for semantic queries

Source: `books_metadata_cleaned.csv` — **10,170 books**.

**74 categories:** 58 subjects, 15 book forms and 1 audience category.

This is a curated vocabulary grounded in catalogue titles and, where needed, book flaps. The source has no category column. Categories may overlap; the examples support their presence, but books have not been fully classified.

## Using the list

- Use `data/book_categories.json` for structured categories, bilingual labels, aliases, scope descriptions, query examples and source evidence.
- Use `data/book_categories.csv` for spreadsheet review. List cells contain JSON arrays.
- Embed each category's `embedding_text` using the same multilingual embedding model used for the query, then retrieve relevant category IDs as semantic hints.
- Match a book using its title and flap. Keep its subject, form and audience as separate tags, and allow multiple tags per facet.
- Retrieve and rank actual books using their metadata. Use category matches to assist retrieval; broad categories can miss distinctions within a subject.
- Keep author, place, historical period and publication year as separate constraints. Publication year does not identify the era the book discusses.

Example: **কিশোরদের জন্য নারী মুক্তিযোদ্ধাদের স্মৃতিকথা** combines `war_women` (subject), `autobiography_memoir` (form) and `children_young_adults` (audience). This illustrates query decomposition, not a guarantee that a book satisfying every constraint exists.

The exports are reference data; the current search application does not load them automatically. The existing `data/taxonomy.yaml` is unchanged.

## মুক্তিযুদ্ধ / Bangladesh Liberation War

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `liberation_war` | মুক্তিযুদ্ধের ইতিহাস / Liberation War history | The origins, course and aftermath of Bangladesh's 1971 independence war. | বাংলাদেশের মুক্তিযুদ্ধের পটভূমি নিয়ে বই | বাংলাদেশের স্বাধীনতা যুদ্ধ ও গণহত্যার ইতিহাস — লেফটেন্যান্ট ইমরান আহমেদ চৌধুরী বিইএম (record 1) |
| `war_campaigns` | রণাঙ্গন ও গেরিলা যুদ্ধ / Military campaigns and guerrilla warfare | Battlefields, military operations, sectors, commanders and guerrilla resistance in 1971. | একাত্তরের গেরিলা অভিযান ও সেক্টর কমান্ডারদের নিয়ে বই | একাত্তরের মুক্তিযুদ্ধ : একজন সেক্টর কমান্ডারের স্মৃতিকথা — কর্নেল (অব.) কাজী নূর-উজ্জামান (record 274) |
| `war_atrocities` | গণহত্যা ও যুদ্ধাপরাধ / Genocide and war crimes | Mass killings, torture, killing sites, collaborators and accountability, especially in 1971. | একাত্তরের গণহত্যা ও যুদ্ধাপরাধের বিচার নিয়ে বই | একাত্তরের গণহত্যা, নির্যাতন ও যুদ্ধাপরাধীদের বিচার — শাহরিয়ার কবির (record 157) |
| `war_refugees` | যুদ্ধকালীন শরণার্থী ও বাস্তুচ্যুতি / Wartime refugees and displacement | Refugee camps, flight, exile and civilian displacement during the Liberation War. | মুক্তিযুদ্ধের সময় শরণার্থী শিবিরের জীবন কেমন ছিল | ১৯৭১ শরণার্থী জীবনের স্মৃতি — প্রফেসর ননী গোপাল সরকার (record 530) |
| `war_women` | মুক্তিযুদ্ধে নারী ও বীরাঙ্গনা / Women in the Liberation War | Women fighters, survivors, wartime sexual violence and women's experiences of 1971. | নারী মুক্তিযোদ্ধা ও বীরাঙ্গনাদের অভিজ্ঞতা নিয়ে বই | গেরিলা নারী : নারী মুক্তিযোদ্ধাদের স্মৃতিকথা — বিচারপতি মুহাম্মদ হাবিবুর রহমান (record 195) |
| `war_documents` | মুক্তিযুদ্ধের দলিল ও আন্তর্জাতিক প্রতিক্রিয়া / War documents and international responses | Primary documents, diplomatic records and foreign reactions concerning Bangladesh's independence war. | বিদেশি দলিলে বাংলাদেশের মুক্তিযুদ্ধ সম্পর্কে কী আছে | মুক্তিযুদ্ধের দলিলপত্র : ফরিদপুর — আবু সাঈদ খান (record 312) |
| `war_media` | মুক্তিযুদ্ধের বেতার ও সাংস্কৃতিক প্রতিরোধ / Wartime broadcasting and cultural resistance | Swadhin Bangla Betar Kendra, wartime songs and cultural mobilisation for independence. | স্বাধীন বাংলা বেতার কেন্দ্রের ভূমিকা নিয়ে বই | গণসঙ্গীত ও মুক্তিযুদ্ধ — ফকির আলমগীর (record 526) |

## ইতিহাস ও ঐতিহ্য / History and heritage

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `national_history` | বাংলাদেশ ও বাংলার ইতিহাস / History of Bangladesh and Bengal | The historical development of Bengal, Bengali identity and Bangladesh across periods. | বাঙালি জাতির পরিচয় ও বাংলাদেশের ইতিহাস নিয়ে বই | বাংলাদেশের ইতিহাসের রূপরেখা — আশফাক হোসেন (record 2788) |
| `local_history` | আঞ্চলিক ও নগর ইতিহাস / Regional and urban history | The histories of districts, towns, cities and local communities, including Dhaka. | পুরনো ঢাকার সমাজ ও নগরজীবন নিয়ে বই | ঢাকার ইতিহাস — শ্রীকেদারনাথ মজুমদার (record 2563) |
| `ancient_civilizations` | প্রাচীন ইতিহাস ও সভ্যতা / Ancient history and civilizations | Ancient societies and civilizations, including Egypt, Greece, Rome and the Indus region. | প্রাচীন মিশর ও সিন্ধু সভ্যতা সম্পর্কে বই | রোমান সাম্রাজ্য — সাহাদত হোসেন খান (record 2729) |
| `archaeology_architecture` | প্রত্নতত্ত্ব ও স্থাপত্য ঐতিহ্য / Archaeology and architectural heritage | Archaeological sites, material remains, historical buildings, temples and mosques. | বাংলার প্রাচীন মন্দির ও মসজিদের স্থাপত্য নিয়ে বই | বাংলাদেশের প্রত্ন স্থাপত্য — নওশের আলী হিরা (record 2952) |
| `medieval_empires` | মধ্যযুগ, সুলতানি ও মুঘল ইতিহাস / Medieval kingdoms and empires | Medieval dynasties, sultanates, Mughal rule and other historical empires. | বাংলায় সুলতানি ও মুঘল শাসন নিয়ে বই | মুঘল ভারতের ইতিহাস — ড. মোহাম্মদ ছিদ্দিকুর রহমান খান (record 2666) |
| `colonial_history` | ঔপনিবেশিক শাসন ও প্রতিরোধ / Colonial rule and anticolonial resistance | British rule, the East India Company, colonial institutions and resistance movements. | ব্রিটিশ আমলে বাংলার কৃষক বিদ্রোহ নিয়ে বই | ইস্ট ইন্ডিয়া কোম্পানির অর্থনৈতিক ইতিহাস : ভারত উপমহাদেশ — তীর্থংকর রায় (record 4490) |
| `bengal_partition` | বঙ্গভঙ্গ ও স্বদেশী আন্দোলন / Partition of Bengal and the Swadeshi movement | The 1905 partition of Bengal, its political context and related resistance. | ১৯০৫ সালের বঙ্গভঙ্গ ও স্বদেশী আন্দোলন নিয়ে বই | বঙ্গভঙ্গ ও তৎকাল — গোলাম মুস্তাফা (record 4013) |
| `india_partition` | দেশভাগ ও সাম্প্রদায়িকতা / Partition of India and communalism | The 1947 partition, communal conflict, divided communities and its human consequences. | দেশভাগে সাধারণ মানুষের জীবন কীভাবে বদলে গেল | সাতচল্লিশের দেশভাগে গান্ধী ও জিন্নাহ — সিরাজুল ইসলাম চৌধুরী (record 2393) |
| `world_history` | বিশ্ব ইতিহাস ও বিশ্বযুদ্ধ / World history and world wars | Histories of other countries, global conflicts and major international historical developments. | দ্বিতীয় বিশ্বযুদ্ধ ও ইউরোপের ইতিহাস নিয়ে বই | বিশ্ব ইতিহাস প্রসঙ্গ — জওহরলাল নেহেরু (record 2329) |

## রাজনীতি ও রাষ্ট্র / Politics and government

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `language_movement` | ভাষা আন্দোলন / Bengali Language Movement | The struggle for Bengali language rights, 1952 and regional language movements. | বিভিন্ন জেলায় ভাষা আন্দোলনের ইতিহাস নিয়ে বই | ভাষা আন্দোলনের কথা — ড. গুলশান আরা (record 3656) |
| `pakistan_politics` | পাকিস্তান আমলের রাজনীতি / Politics in the Pakistan period | East Pakistan politics, the Six Points, autonomy movements and political conflict before independence. | ছয় দফা আন্দোলন ও পূর্ব পাকিস্তানের রাজনীতি নিয়ে বই | পূর্ব পাকিস্তানের জন্ম থেকে মৃত্যু — জিবলু রহমান (record 2615) |
| `bangladesh_politics` | স্বাধীন বাংলাদেশের রাজনীতি / Politics of independent Bangladesh | Political parties, leadership, changes of government and political crises after independence. | স্বাধীনতার পর বাংলাদেশের রাজনৈতিক পরিবর্তন নিয়ে বই | বাংলাদেশের রাজনীতিতে জাসদ — জিয়াউল হক মুক্তা (record 5444) |
| `democratic_movements` | ছাত্র আন্দোলন ও গণঅভ্যুত্থান / Student movements and mass uprisings | Student politics, popular protest, resistance to authoritarian rule and mass uprisings. | ছাত্র আন্দোলন ও স্বৈরাচারবিরোধী সংগ্রাম নিয়ে বই | বাংলাদেশের ছাত্র আন্দোলনের ইতিহাস — আমজাদ হোসেন (record 3409) |
| `july_uprising` | জুলাই-আগস্ট ২০২৪ আন্দোলন / July-August 2024 movement | Catalogue accounts of the July-August 2024 protests, uprising and participants' experiences. | জুলাই-আগস্ট ২০২৪ আন্দোলনের প্রত্যক্ষ অভিজ্ঞতা নিয়ে বই | কোটা আন্দোলন থেকে গণঅভ্যুত্থান — নজরুল ইসলাম (record 2479) |
| `governance_democracy` | গণতন্ত্র, সংবিধান ও শাসনব্যবস্থা / Democracy, constitutions and governance | Democratic institutions, elections, constitutional debates, public administration and state organisation. | বাংলাদেশের সংবিধান ও নির্বাচনি ব্যবস্থা নিয়ে বই | গণতন্ত্রের ১৫ বছর — মতিউর রহমান (record 4723) |
| `international_relations` | কূটনীতি ও আন্তর্জাতিক সম্পর্ক / Diplomacy and international relations | Foreign policy, diplomatic practice and relations among states. | বাংলাদেশের পররাষ্ট্রনীতি ও কূটনীতি নিয়ে বই | The A to Z Of আন্তর্জাতিক সম্পর্ক — Noore Alam Siddiqui (record 5907) |
| `political_ideologies` | রাজনৈতিক মতবাদ ও বিপ্লব / Political ideologies and revolutions | Marxism, socialism, capitalism, nationalism and revolutionary thought and movements. | পুঁজিবাদ ও সমাজতন্ত্রের তুলনা নিয়ে বই | সমাজতন্ত্রের অনিবার্য ভবিষ্যৎ — বদরুদ্দীন উমর (record 5430) |

## সমাজ ও সংস্কৃতি / Society and culture

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `social_anthropology` | সমাজবিজ্ঞান ও নৃবিজ্ঞান / Sociology and anthropology | Social structures, communities, cultural practices and anthropological studies. | দক্ষিণ এশিয়ার সমাজ ও সংস্কৃতি নিয়ে নৃবিজ্ঞানের বই | ফরাসি সমাজবিজ্ঞান — সাদাত উল্লাহ খান (record 3195) |
| `folk_culture` | লোকসংস্কৃতি ও সাংস্কৃতিক ঐতিহ্য / Folklore and cultural heritage | Folk traditions, local customs, crafts, cultural identity and intangible heritage. | বাংলাদেশের লোকসংস্কৃতি ও লোকজ ঐতিহ্য নিয়ে বই | লোকসংস্কৃতির কথকতা — চন্দন চৌধুরী (record 4846) |
| `women_gender` | নারী, অধিকার ও সমাজ / Women, rights and society | Women's lives, social roles, education, rights and gender-related struggles. | নারী শিক্ষা ও সমাজে নারীর অবস্থান নিয়ে বই | নারীমুক্তির রাজনৈতিক সংগ্রাম — জলি তালুকদার (record 5588) |
| `ethnic_indigenous` | আদিবাসী ও জাতিগত জনগোষ্ঠী / Indigenous and ethnic communities | Indigenous communities, ethnic identities, minority experiences and the Chittagong Hill Tracts. | পার্বত্য চট্টগ্রামের আদিবাসীদের ইতিহাস ও জীবন নিয়ে বই | পার্বত্য চট্টগ্রামের ইতিহাস — জামাল উদ্দিন (record 4491) |
| `journalism` | সাংবাদিকতা ও গণমাধ্যম / Journalism and media | Newspapers, reporting, media history and journalists' professional experiences. | বাংলা সংবাদপত্র ও সাংবাদিকতার ইতিহাস নিয়ে বই | সাংবাদিকতায় চার দশক — মাহবুব আলম (record 9227) |

## ধর্ম ও ইসলামি অধ্যয়ন / Religion and Islamic studies

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `islamic_history` | ইসলামি ইতিহাস ও মুসলিম সভ্যতা / Islamic history and Muslim civilization | Muslim societies, Islamic dynasties, cultural achievements and historical developments. | মুসলিম সভ্যতা ও ইসলামি ইতিহাস নিয়ে বই | ইসলামের ইতিহাসের মহাবীর খালিদ বিন ওয়ালিদ (রা) — মোঃ সাইফুল আনোয়ার (record 8602) |
| `prophet_biography` | নবীজীবন ও সিরাত / Prophets' lives and Sirah | Lives of prophets, particularly the life, character and mission of Prophet Muhammad. | নবী মুহাম্মদের জীবন ও চরিত্র নিয়ে বই | নবি জীবনের গল্প — আরিফ আজাদ (record 7561) |
| `companions_scholars` | সাহাবি ও ইসলামি ব্যক্তিত্ব / Companions and Islamic figures | Lives and contributions of the Companions, caliphs, scholars and notable Islamic figures. | সাহাবিদের জীবন ও জ্ঞানচর্চা নিয়ে বই | সাহাবিদের কারামত — নাজিবুল্লাহ সিদ্দিকী (record 7928) |
| `islamic_teachings` | কুরআন, হাদিস ও ইসলামি জীবন / Quran, Hadith and Islamic life | Islamic scripture, Hadith, ethics, faith and religious practice. | কুরআন ও হাদিসের আলোকে জীবন ও নৈতিকতা নিয়ে বই | কুরআনের সৌন্দর্যে অভিভূত — নোমান আলী খান (record 7715) |
| `comparative_religion` | ধর্মের ইতিহাস ও তুলনামূলক ধর্ম / Religious history and comparative religion | Histories, beliefs and traditions across religions, including Hinduism and Buddhism. | হিন্দু ও বৌদ্ধ ধর্মের ইতিহাস সম্পর্কে বই | হিন্দুধর্ম — ক্ষিতিমোহন সেন (record 2304) |
| `sufism` | সুফিবাদ ও মরমি ঐতিহ্য / Sufism and mystical traditions | Sufi thought, saints, shrines and mystical traditions in Bengal and elsewhere. | বাংলার সুফি সাধক ও মরমি ঐতিহ্য নিয়ে বই | মাজার সংস্কৃতির বহুমাত্রিক প্রভাব — ড. মো. আশ্রাফুল করিম (record 7393) |

## ভাষা ও সাহিত্য / Language and literature

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `language_literature` | ভাষা, সাহিত্য ইতিহাস ও সমালোচনা / Language, literary history and criticism | Languages, scripts, literary development, interpretation and criticism of literary works. | বাংলা সাহিত্যের ইতিহাস ও সাহিত্য সমালোচনা নিয়ে বই | বাংলা সাহিত্যের ইতিহাস — মোতাহার হোসেন সুফী (record 2495) |
| `author_studies` | রবীন্দ্রনাথ, নজরুল ও সাহিত্যিকদের জীবন-কর্ম / Writers' lives and works | Studies of writers and poets, their biographies, ideas, literary works and correspondence. | রবীন্দ্রনাথ ও নজরুলের জীবন ও সাহিত্যকর্ম নিয়ে বই | মানিক বন্দ্যোপাধ্যায় — অঞ্জন আচার্য (record 8409) |

## শিল্প ও বিনোদন / Arts and entertainment

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `cinema` | চলচ্চিত্র ও চলচ্চিত্র নির্মাণ / Cinema and filmmaking | Film history, criticism, filmmaking, directors and cinema culture. | বাংলাদেশের চলচ্চিত্রের ইতিহাস ও নির্মাণশৈলী নিয়ে বই | ফিল্মমেকারের ভাষা (ইরানি) — বিজয় আহমেদ (record 8513) |
| `theatre` | নাট্যকলা ও মঞ্চ / Theatre and performance | Theatre history, dramatic art, staging and performance traditions; use a form tag separately for play texts. | নাট্যকলার ইতিহাস ও মঞ্চচর্চা নিয়ে বই | ইউরোপীয় নাটক ইতিহাস ও গল্পকথা — অমিতাভ চৌধুরী (record 3959) |
| `music` | সংগীত, গান ও সংগীতশিল্পী / Music, songs and musicians | Music history, folk and classical music, notation, song collections and musicians' lives. | বাংলার লোকগান ও সংগীতশিল্পীদের জীবন নিয়ে বই | স্বরবিতান (প্রথম খন্ড) — রবীন্দ্রনাথ ঠাকুর (record 6900) |
| `visual_arts` | চিত্রকলা, ভাস্কর্য ও আলোকচিত্র / Visual arts and photography | Painting, sculpture, art history, visual culture and photography. | বাংলাদেশের চিত্রকলা ও আলোকচিত্র নিয়ে বই | আলোকচিত্রালাপ : পাভেল রহমানের সঙ্গে — সাহাদাত পারভেজ (record 9405) |

## ভ্রমণ ও প্রবাস / Travel and migration

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `travel_writing` | দেশ-বিদেশ ভ্রমণ / Travel and exploration | Journeys, destinations, observations and travel experiences in Bangladesh and abroad. | দেশ-বিদেশের মানুষ ও সংস্কৃতি নিয়ে ভ্রমণকাহিনি | অন্যরকম ভ্রমণ — সমরেশ মজুমদার (record 5937) |
| `migration_diaspora` | প্রবাস, অভিবাসন ও প্রবাসজীবন / Migration, diaspora and life abroad | Migration, expatriate lives, settlement abroad and diaspora experiences. | প্রবাসে বাঙালিদের জীবন ও অভিজ্ঞতা নিয়ে বই | কানাডা অভিবাসনের আবেদন প্রত্যাশা ও বাস্তবতা — এম এল গনি (record 6511) |
| `pilgrimage` | হজ ও তীর্থভ্রমণ / Hajj and pilgrimage travel | Religious journeys, Hajj experiences and visits to pilgrimage destinations. | হজ ও ধর্মীয় তীর্থভ্রমণের অভিজ্ঞতা নিয়ে বই | হিন্দুতীর্থ কামরূপ কামাখ্যা — অরুণ মুখোপাধ্যায় (record 6026) |

## অর্থনীতি ও ব্যবসা / Economics and business

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `economics_development` | অর্থনীতি ও উন্নয়ন / Economics and development | Economic theory, political economy, poverty, public finance and economic development. | বাংলাদেশের অর্থনীতি ও দারিদ্র্য নিয়ে বই | দারিদ্র্যের অর্থনীতি : অতীত, বর্তমান ও ভবিষ্যৎ — আকবর আলি খান (record 9666) |
| `islamic_finance` | ইসলামি অর্থনীতি ও ব্যবসায় নীতি / Islamic economics and business ethics | Islamic economic principles, interest-free finance, Islamic banking and religious business ethics. | সুদমুক্ত অর্থায়ন ও ইসলামি ব্যবসায় নীতি নিয়ে বই | ইসলামী অর্থনীতি  —  পুঁজিবাদী অর্থনীতির বিকল্প প্রস্তাবনা (record 9762) |
| `banking_finance` | ব্যাংকিং, বীমা ও আর্থিক ব্যবস্থা / Banking, insurance and financial systems | Banks, banking practice, insurance, financial institutions and trade finance. | ব্যাংকিং ও আন্তর্জাতিক বাণিজ্য অর্থায়ন নিয়ে বই | মানবিক ব্যাংকিং — আতিউর রহমান (record 10029) |
| `personal_finance` | ব্যক্তিগত অর্থব্যবস্থা ও সঞ্চয় / Personal finance and money management | Managing personal money, savings, budgeting and financial habits. | নিজের টাকা ও সঞ্চয় ভালোভাবে পরিচালনা করার বই | দি আর্ট অব পারসোনাল ফাইন্যান্স ম্যানেজমেন্ট — সাইফুল হোসেন (record 9630) |
| `investment` | বিনিয়োগ ও শেয়ারবাজার / Investment and stock markets | Investment principles, shares, capital markets and market participation. | শেয়ারবাজার ও বিনিয়োগের মৌলিক বিষয় নিয়ে বই | পুঁজিবাজারের অর্থ বৎসরের ক্যালেন্ডার ও ইত্যাদি — মোহাম্মদ মহিউদ্দিন এফ. সি. এম. এ,সায়রা বানু (এসি এমএ) (record 9743) |
| `marketing_sales` | মার্কেটিং, ব্র্যান্ডিং ও বিক্রয় / Marketing, branding and sales | Marketing strategy, brand building, selling, digital promotion and customer relationships. | ডিজিটাল মার্কেটিং ও বিক্রয় বাড়ানোর কৌশল নিয়ে বই | ইমোশনাল ব্র্যান্ডিং — মোঃ মাছুম চৌধুরী (record 9827) |
| `entrepreneurship` | উদ্যোক্তা, ব্যবসা ও ই-কমার্স / Entrepreneurship, business and e-commerce | Starting and operating businesses, entrepreneurial thinking, online commerce and startups. | নতুন উদ্যোক্তাদের অনলাইন ব্যবসা শুরু করার বই | উদ্যোক্তাদের হিসাববিজ্ঞান — মঈন রেজা নাদিম (record 10088) |
| `management_leadership` | ব্যবস্থাপনা ও নেতৃত্ব / Management and leadership | Managing organisations and teams, leadership, strategy and organisational practice. | দল পরিচালনা ও নেতৃত্বের দক্ষতা বাড়ানোর বই | ম্যানেজমেন্ট — ব্রায়ান ট্রেসি (record 9684) |
| `career_productivity` | ক্যারিয়ার, দক্ষতা ও ব্যক্তিগত উন্নতি / Career, skills and personal development | Professional growth, success habits, productivity, communication and customer service. | ক্যারিয়ার ও কাস্টমার সার্ভিসের দক্ষতা বাড়ানোর বই | কাস্টমার সার্ভিসের কলাকৌশল — কয়েস সামী (record 10081) |

## জ্ঞান, শিক্ষা ও জীবন / Knowledge, education and life

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `philosophy` | দর্শন ও চিন্তাচর্চা / Philosophy and intellectual thought | Philosophical ideas, thinkers, ethics, reasoning and intellectual history. | দর্শন ও দার্শনিকদের চিন্তা সম্পর্কে বই | সেইসব দার্শনিক — সরদার ফজলুল করিম (record 7683) |
| `science` | বিজ্ঞান ও বৈজ্ঞানিক চিন্তা / Science and scientific thought | Scientific ideas, discoveries, evolution and the lives and work of scientists. | বিবর্তন ও বিজ্ঞানের ইতিহাস নিয়ে বই | জৈববিবর্তনবাদ: দেড়শ’ বছরের দ্বন্দ বিরোধ — প্রকৌশলী মোঃ মনিরুল ইসলাম (record 4257) |
| `education` | শিক্ষা ও শিক্ষাপ্রতিষ্ঠান / Education and educational institutions | Educational thought, schools, universities, teaching and institutional histories. | বাংলাদেশের বিশ্ববিদ্যালয় ও শিক্ষাব্যবস্থার ইতিহাস নিয়ে বই | বিশ্ববিদ্যালয়ের ইতিহাস: আদিপর্ব — শিশির ভট্টাচার্য্য (record 4014) |
| `health_medicine` | স্বাস্থ্য, চিকিৎসা ও জনস্বাস্থ্য / Health, medicine and public health | Medical experiences, healthcare, public health movements and medical figures. | জনস্বাস্থ্য আন্দোলন ও চিকিৎসকদের অভিজ্ঞতা নিয়ে বই | স্বাস্থ্য অর্থনীতি ও আমাদের উন্নয়ন ভাবনা — শরীফ নাফে আচ্ছাবের (record 9912) |
| `sports` | ক্রীড়া ও ক্রীড়াব্যক্তিত্ব / Sports and sporting figures | Sporting lives, athletes, sports personalities and sports culture. | ক্রীড়াব্যক্তিত্বদের জীবন ও অভিজ্ঞতা নিয়ে বই | সেরা ১০০ ক্রীড়াব্যাক্তিত্ব — হাসান শরীফ (record 8496) |

## বইয়ের ধরন / Book forms

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `history_form` | ইতিহাসগ্রন্থ / Historical accounts | Books presented as historical accounts; combine with a subject, place or period. | বাংলার ইতিহাসের বই | বাংলাদেশের ইতিহাসের রূপরেখা — আশফাক হোসেন (record 2788) |
| `biography` | জীবনী / Biography | Accounts of another person's life, distinct from first-person autobiography. | একজন বিজ্ঞানীর জীবনী | আহমদ শরীফ : জীবন ও কর্ম — মাসুদ রহমান (record 8616) |
| `autobiography_memoir` | আত্মজীবনী ও স্মৃতিকথা / Autobiography and memoir | First-person life narratives, recollections and memories of particular experiences. | একজন মুক্তিযোদ্ধার স্মৃতিকথা | একাত্তরের স্মৃতিকথা — মোহাম্মদ মতিউর রহমান সেলিম (record 3) |
| `diary` | ডায়েরি ও দিনলিপি / Diaries and journals | Dated journals, diaries and day-by-day records of personal or historical experience. | একাত্তরের যুদ্ধদিনের ডায়েরি | ৭১ এর রোজনামচা — আহসান হাবীব (কার্টুনিস্ট) (record 284) |
| `novel` | উপন্যাস / Novels | Long-form fictional narratives; historical subject matter does not make them factual accounts. | মুক্তিযুদ্ধের পটভূমিতে লেখা উপন্যাস | মুক্তিযুদ্ধের উপন্যাস - মুক্তি — শাহআলম সাজু (record 20) |
| `short_story` | গল্প ও ছোটগল্প / Stories and short fiction | Stories and short-story collections; narrative nonfiction can also use the word story, so verify the flap. | দেশভাগ নিয়ে ছোটগল্প | গল্পগুলো বিজয়ের — ফখরুল হাসান (record 1444) |
| `poetry` | কবিতা, কাব্য ও ছড়া / Poetry and rhymes | Poems, verse, poetry collections and rhymes. | স্বাধীনতা নিয়ে কবিতা ও ছড়া | মুক্তিযুদ্ধের কবিতা — আবুল হাসনাত (record 212) |
| `essay` | প্রবন্ধ ও নিবন্ধ / Essays and articles | Essays, analytical articles, columns and collected prose commentary. | সমাজ ও সংস্কৃতি নিয়ে প্রবন্ধ | সময়ের মুখোমুখি : বামপন্থা, মক্তিযুদ্ধ ও বুদ্ধিজীবীদের ভূমিকা সম্পর্কে বিতর্কমূলক নিবন্ধের সংকলন — হাসান শফি (record 835) |
| `research` | গবেষণা ও বিশ্লেষণ / Research and analytical studies | Research-based studies, theses and analytical investigations of a topic. | ভাষা আন্দোলন নিয়ে গবেষণাগ্রন্থ | তাত্ত্বিক বিশ্লেষণে আমাদের স্বাধীনতা সংগ্রাম — ওমর খালেদ রুমি (record 1050) |
| `reference` | অভিধান, তথ্যকোষ ও রেফারেন্স / Dictionaries, encyclopedias and reference | Works designed for consultation, including dictionaries, encyclopedias and glossaries. | অর্থনীতির পরিভাষা শেখার অভিধান | ক্রুসেড বিশ্বকোষ — ড. আলী মুহাম্মদ সাল্লাবি (record 4432) |
| `interview` | সাক্ষাৎকার / Interviews | Interviews, conversations and question-and-answer collections. | চলচ্চিত্র নির্মাতাদের সাক্ষাৎকার | অর্ধশত সাক্ষাৎকারে আল মাহমুদ — সরদার আবদুর রহমান (record 8749) |
| `letters` | চিঠিপত্র / Letters and correspondence | Personal letters, correspondence and edited collections of letters. | রবীন্দ্রনাথের চিঠিপত্র | সুকান্তের গল্পসমগ্র ও পত্রগুচ্ছ — সুকান্ত ভট্টাচার্য (record 8571) |
| `anthology` | সংকলন ও রচনাসমগ্র / Anthologies and collected works | Selected or collected writings, multi-author anthologies and complete works. | মুক্তিযুদ্ধের নির্বাচিত গল্পের সংকলন | নির্বাচিত কলাম — আহমদ রফিক (record 1860) |
| `translation` | অনুবাদ / Translations | Works translated from another language; language of origin should be verified separately. | বাংলায় অনূদিত বিশ্ব ইতিহাসের বই | অনূদিত কথাসমগ্র : বিশ্ববরেণ্য ২৫ জনের সাক্ষাৎকার — রাজু আলাউদ্দিন (record 9170) |
| `play` | নাটক / Plays | Dramatic texts and scripts; distinguish these from books analysing theatre. | মুক্তিযুদ্ধ নিয়ে লেখা নাটক | মুক্তিযুদ্ধের তিন নাটক — আবদুল মান্নান (record 1311) |

## পাঠকগোষ্ঠী / Audiences

| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |
|---|---|---|---|---|
| `children_young_adults` | শিশু-কিশোর / Children and young adults | Books explicitly intended for children or young readers; combine with subject and form. | কিশোরদের জন্য মুক্তিযুদ্ধের গল্প | শিশু-কিশোরদের নজরুল — সাকিব জামাল (record 9060) |

## Rebuild and evidence notes

Run `python scripts/build_book_categories.py` after catalogue changes. Edit the curated definitions in that script to change category scope or aliases.

`source_record` is the 1-based data-record index (excluding the header), not a physical CSV line number or the search application's book ID. `title_evidence_book_count` counts literal evidence-term matches in titles; it is not a semantic category membership count. Flaps are checked only if a category has no title evidence. Author biographies are excluded because an author's interests do not establish a book's subject.

The JSON records a SHA-256 fingerprint of the source file for reproducibility. Example queries illustrate intent; retrieval performance has not been evaluated.
