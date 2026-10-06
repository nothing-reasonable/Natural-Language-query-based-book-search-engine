"""Export a curated semantic category list with evidence from the source catalogue.

Standard library only. This builds a vocabulary, not per-book classifications.
Run: python scripts/build_book_categories.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "liberation": ("মুক্তিযুদ্ধ", "Bangladesh Liberation War"),
    "history": ("ইতিহাস ও ঐতিহ্য", "History and heritage"),
    "politics": ("রাজনীতি ও রাষ্ট্র", "Politics and government"),
    "society": ("সমাজ ও সংস্কৃতি", "Society and culture"),
    "religion": ("ধর্ম ও ইসলামি অধ্যয়ন", "Religion and Islamic studies"),
    "literature": ("ভাষা ও সাহিত্য", "Language and literature"),
    "arts": ("শিল্প ও বিনোদন", "Arts and entertainment"),
    "travel": ("ভ্রমণ ও প্রবাস", "Travel and migration"),
    "business": ("অর্থনীতি ও ব্যবসা", "Economics and business"),
    "knowledge": ("জ্ঞান, শিক্ষা ও জীবন", "Knowledge, education and life"),
    "formats": ("বইয়ের ধরন", "Book forms"),
    "audiences": ("পাঠকগোষ্ঠী", "Audiences"),
}

# Titles reviewed for clearer examples where a broad word can identify a book
# about a form rather than a work in that form (e.g. criticism about novels).
PREFERRED_TITLES = {
    "liberation_war": ["বাংলাদেশের স্বাধীনতা যুদ্ধ ও গণহত্যার ইতিহাস"],
    "war_atrocities": ["একাত্তরের গণহত্যা, নির্যাতন ও যুদ্ধাপরাধীদের বিচার"],
    "local_history": ["ঢাকার ইতিহাস", "চট্টগ্রামের ইতিহাস", "ফরিদপুরের ইতিহাস"],
    "medieval_empires": ["মুঘল ভারতের ইতিহাস", "মধ্যযুগের বাংলা : সমাজ ও সংস্কৃতি",
                         "আব্বাসীয় খিলাফত ও আঞ্চলিক রাজবংশসমূহ (৭৫০-১২৫৮)"],
    "india_partition": ["সাতচল্লিশের দেশভাগে গান্ধী ও জিন্নাহ",
                        "জিন্না: ভারত দেশভাগ স্বাধীনতা", "দেশভাগের গল্প"],
    "travel_writing": ["অন্যরকম ভ্রমণ", "ভ্রমণসমগ্র"],
    "migration_diaspora": ["কানাডা অভিবাসনের আবেদন প্রত্যাশা ও বাস্তবতা",
                           "প্রবাসে মুক্তিযুদ্ধের দিনগুলি"],
    "personal_finance": ["দি আর্ট অব পারসোনাল ফাইন্যান্স ম্যানেজমেন্ট",
                         "দ্য নিটি গ্রিটি অব ফাইন্যান্স ম্যানেজমেন্ট ইন লাইফ"],
    "management_leadership": ["ম্যানেজমেন্ট", "প্রজেক্ট ম্যানেজমেন্টের খুঁটিনাটি"],
    "history_form": ["বাংলাদেশের ইতিহাসের রূপরেখা", "মুঘল ভারতের ইতিহাস"],
    "biography": ["আহমদ শরীফ : জীবন ও কর্ম", "জীবন ও কর্ম : আয়েশা রাযি."],
    "autobiography_memoir": ["একাত্তরের স্মৃতিকথা",
                            "গেরিলা নারী : নারী মুক্তিযোদ্ধাদের স্মৃতিকথা",
                            "একাত্তরের মুক্তিযুদ্ধ : একজন সেক্টর কমান্ডারের স্মৃতিকথা"],
    "novel": ["মুক্তিযুদ্ধের উপন্যাস - মুক্তি", "সেরা পাঁচ মুক্তিযুদ্ধের উপন্যাস",
              "মুক্তিযুদ্ধের কিশোর উপন্যাস রহস্যময় অরণ্যে একদিন"],
    "poetry": ["মুক্তিযুদ্ধের কবিতা", "মুক্তিযুদ্ধের ছড়া কবিতা",
               "একাত্তরে রচিত মুক্তিযুদ্ধের কবিতা"],
}

# id | facet | group | Bengali label | English label | scope | aliases (;) |
# catalogue evidence terms (;) | Bengali example query | English example query
# Evidence terms are intentionally narrower than query aliases. A term in a title
# supports the category's existence; it does not prove the book's classification.
DEFINITIONS = """
liberation_war|subject|liberation|মুক্তিযুদ্ধের ইতিহাস|Liberation War history|The origins, course and aftermath of Bangladesh's 1971 independence war.|মুক্তিযুদ্ধ;একাত্তর;বাংলাদেশের স্বাধীনতা যুদ্ধ;1971 liberation war;muktijuddho|মুক্তিযুদ্ধ;একাত্তর;স্বাধীনতা যুদ্ধ|বাংলাদেশের মুক্তিযুদ্ধের পটভূমি নিয়ে বই|Books about the origins of Bangladesh's Liberation War
war_campaigns|subject|liberation|রণাঙ্গন ও গেরিলা যুদ্ধ|Military campaigns and guerrilla warfare|Battlefields, military operations, sectors, commanders and guerrilla resistance in 1971.|রণাঙ্গন;গেরিলা;মুক্তিবাহিনী;সেক্টর কমান্ডার;guerrilla warfare;battlefield|রণাঙ্গন;গেরিলা;সেক্টর কমান্ডার|একাত্তরের গেরিলা অভিযান ও সেক্টর কমান্ডারদের নিয়ে বই|Books about guerrilla operations and sector commanders in 1971
war_atrocities|subject|liberation|গণহত্যা ও যুদ্ধাপরাধ|Genocide and war crimes|Mass killings, torture, killing sites, collaborators and accountability, especially in 1971.|গণহত্যা;বধ্যভূমি;যুদ্ধাপরাধ;রাজাকার;genocide;war crimes|গণহত্যা;বধ্যভূমি;যুদ্ধাপরাধ|একাত্তরের গণহত্যা ও যুদ্ধাপরাধের বিচার নিয়ে বই|Books about the 1971 genocide and war crimes trials
war_refugees|subject|liberation|যুদ্ধকালীন শরণার্থী ও বাস্তুচ্যুতি|Wartime refugees and displacement|Refugee camps, flight, exile and civilian displacement during the Liberation War.|শরণার্থী;শরণার্থী শিবির;উদ্বাস্তু;refugee camps;wartime displacement|শরণার্থী শিবির;শরণার্থী জীবনের;শরণার্থী|মুক্তিযুদ্ধের সময় শরণার্থী শিবিরের জীবন কেমন ছিল|Books about life in refugee camps during the Liberation War
war_women|subject|liberation|মুক্তিযুদ্ধে নারী ও বীরাঙ্গনা|Women in the Liberation War|Women fighters, survivors, wartime sexual violence and women's experiences of 1971.|বীরাঙ্গনা;নারী মুক্তিযোদ্ধা;মুক্তিযুদ্ধে নারী;women freedom fighters;wartime survivors|বীরাঙ্গনা;নারী মুক্তিযোদ্ধা;মুক্তিযুদ্ধে নারী|নারী মুক্তিযোদ্ধা ও বীরাঙ্গনাদের অভিজ্ঞতা নিয়ে বই|Books about women freedom fighters and survivors of 1971
war_documents|subject|liberation|মুক্তিযুদ্ধের দলিল ও আন্তর্জাতিক প্রতিক্রিয়া|War documents and international responses|Primary documents, diplomatic records and foreign reactions concerning Bangladesh's independence war.|মুক্তিযুদ্ধের দলিল;গোপন দলিল;আন্তর্জাতিক প্রতিক্রিয়া;liberation war documents;declassified records|মুক্তিযুদ্ধের দলিল;মুক্তিযুদ্ধের দলিলপত্র;মুক্তিযুদ্ধে মার্কিন;মুক্তিযুদ্ধে আন্তর্জাতিক|বিদেশি দলিলে বাংলাদেশের মুক্তিযুদ্ধ সম্পর্কে কী আছে|Books on foreign records and international responses to the Liberation War
war_media|subject|liberation|মুক্তিযুদ্ধের বেতার ও সাংস্কৃতিক প্রতিরোধ|Wartime broadcasting and cultural resistance|Swadhin Bangla Betar Kendra, wartime songs and cultural mobilisation for independence.|স্বাধীন বাংলা বেতার;মুক্তিযুদ্ধের গান;সাংস্কৃতিক প্রতিরোধ;wartime radio;cultural resistance|স্বাধীন বাংলা বেতার;গণসঙ্গীত ও মুক্তিযুদ্ধ|স্বাধীন বাংলা বেতার কেন্দ্রের ভূমিকা নিয়ে বই|Books about the role of Swadhin Bangla Betar Kendra
national_history|subject|history|বাংলাদেশ ও বাংলার ইতিহাস|History of Bangladesh and Bengal|The historical development of Bengal, Bengali identity and Bangladesh across periods.|বাংলার ইতিহাস;বাংলাদেশের ইতিহাস;বাঙালির ইতিহাস;Bengal history;Bangladesh history|বাংলার ইতিহাস;বাংলাদেশের ইতিহাস;বাঙালির ইতিহাস|বাঙালি জাতির পরিচয় ও বাংলাদেশের ইতিহাস নিয়ে বই|Books about Bengali identity and the history of Bangladesh
local_history|subject|history|আঞ্চলিক ও নগর ইতিহাস|Regional and urban history|The histories of districts, towns, cities and local communities, including Dhaka.|জেলার ইতিহাস;ঢাকার ইতিহাস;আঞ্চলিক ইতিহাস;local history;urban history|ঢাকার ইতিহাস;চট্টগ্রামের ইতিহাস;ফরিদপুরের ইতিহাস;আঞ্চলিক ইতিহাস|পুরনো ঢাকার সমাজ ও নগরজীবন নিয়ে বই|Books about old Dhaka's society and urban life
ancient_civilizations|subject|history|প্রাচীন ইতিহাস ও সভ্যতা|Ancient history and civilizations|Ancient societies and civilizations, including Egypt, Greece, Rome and the Indus region.|প্রাচীন সভ্যতা;মিশরীয় সভ্যতা;সিন্ধু সভ্যতা;ancient civilizations;ancient history|প্রাচীন;মিশরীয় সভ্যতা;সিন্ধু সভ্যতা;রোমান সাম্রাজ্য|প্রাচীন মিশর ও সিন্ধু সভ্যতা সম্পর্কে বই|Books about ancient Egypt and the Indus civilization
archaeology_architecture|subject|history|প্রত্নতত্ত্ব ও স্থাপত্য ঐতিহ্য|Archaeology and architectural heritage|Archaeological sites, material remains, historical buildings, temples and mosques.|প্রত্নতত্ত্ব;প্রত্নস্থাপত্য;পুরাকীর্তি;স্থাপত্য;archaeology;architectural heritage|প্রত্নতাত্ত্বিক;প্রত্ন স্থাপত্য;স্থাপত্য;প্রত্নতত্ত্ব|বাংলার প্রাচীন মন্দির ও মসজিদের স্থাপত্য নিয়ে বই|Books about the architecture of Bengal's old temples and mosques
medieval_empires|subject|history|মধ্যযুগ, সুলতানি ও মুঘল ইতিহাস|Medieval kingdoms and empires|Medieval dynasties, sultanates, Mughal rule and other historical empires.|মধ্যযুগ;সুলতানি আমল;মুঘল;খিলাফত;medieval history;Mughal empire;sultanate|মধ্যযুগ;মুঘল;সুলতান;খিলাফত|বাংলায় সুলতানি ও মুঘল শাসন নিয়ে বই|Books about the sultanates and Mughal rule in Bengal
colonial_history|subject|history|ঔপনিবেশিক শাসন ও প্রতিরোধ|Colonial rule and anticolonial resistance|British rule, the East India Company, colonial institutions and resistance movements.|ব্রিটিশ শাসন;ঔপনিবেশিকতা;ইস্ট ইন্ডিয়া কোম্পানি;নীলবিদ্রোহ;British Raj;anticolonial resistance|ব্রিটিশ;ঔপনিবেশিক;ইস্ট ইন্ডিয়া কোম্পানি;নীলবিদ্রোহ|ব্রিটিশ আমলে বাংলার কৃষক বিদ্রোহ নিয়ে বই|Books about peasant resistance in colonial Bengal
bengal_partition|subject|history|বঙ্গভঙ্গ ও স্বদেশী আন্দোলন|Partition of Bengal and the Swadeshi movement|The 1905 partition of Bengal, its political context and related resistance.|বঙ্গভঙ্গ;১৯০৫;স্বদেশী আন্দোলন;partition of Bengal;Swadeshi movement|বঙ্গভঙ্গ;স্বদেশী আন্দোলন|১৯০৫ সালের বঙ্গভঙ্গ ও স্বদেশী আন্দোলন নিয়ে বই|Books about the 1905 partition of Bengal and the Swadeshi movement
india_partition|subject|history|দেশভাগ ও সাম্প্রদায়িকতা|Partition of India and communalism|The 1947 partition, communal conflict, divided communities and its human consequences.|দেশভাগ;সাতচল্লিশ;সাম্প্রদায়িকতা;দাঙ্গা;partition of India;communal violence|দেশভাগ;সাতচল্লিশ;সাম্প্রদায়িক;সাম্প্রদায়িক|দেশভাগে সাধারণ মানুষের জীবন কীভাবে বদলে গেল|Books about how Partition changed ordinary people's lives
world_history|subject|history|বিশ্ব ইতিহাস ও বিশ্বযুদ্ধ|World history and world wars|Histories of other countries, global conflicts and major international historical developments.|বিশ্ব ইতিহাস;বিশ্বযুদ্ধ;বিশ্বসভ্যতা;world history;world wars|বিশ্ব ইতিহাস;বিশ্বযুদ্ধ;জার্মানির ইতিহাস;চীনের ইতিহাস|দ্বিতীয় বিশ্বযুদ্ধ ও ইউরোপের ইতিহাস নিয়ে বই|Books about World War II and European history
language_movement|subject|politics|ভাষা আন্দোলন|Bengali Language Movement|The struggle for Bengali language rights, 1952 and regional language movements.|ভাষা আন্দোলন;রাষ্ট্রভাষা;বায়ান্ন;একুশে ফেব্রুয়ারি;language movement;language rights|ভাষা আন্দোলন;রাষ্ট্রভাষা;বায়ান্ন;বায়ান্ন|বিভিন্ন জেলায় ভাষা আন্দোলনের ইতিহাস নিয়ে বই|Books about the Language Movement in different districts
pakistan_politics|subject|politics|পাকিস্তান আমলের রাজনীতি|Politics in the Pakistan period|East Pakistan politics, the Six Points, autonomy movements and political conflict before independence.|পূর্ব পাকিস্তান;ছয় দফা;আগরতলা মামলা;আইয়ুব;East Pakistan politics;six point movement|পূর্ব পাকিস্তান;ছয় দফা;ছয় দফা;আগরতলা;আইয়ুব;আইয়ুব|ছয় দফা আন্দোলন ও পূর্ব পাকিস্তানের রাজনীতি নিয়ে বই|Books about the Six Points and politics in East Pakistan
bangladesh_politics|subject|politics|স্বাধীন বাংলাদেশের রাজনীতি|Politics of independent Bangladesh|Political parties, leadership, changes of government and political crises after independence.|বাংলাদেশের রাজনীতি;আওয়ামী লীগ;বিএনপি;পঁচাত্তর;Bangladesh politics;post-independence politics|বাংলাদেশের রাজনীতি;আওয়ামী;আওয়ামী;বিএনপি;পঁচাত্তর|স্বাধীনতার পর বাংলাদেশের রাজনৈতিক পরিবর্তন নিয়ে বই|Books about political change in Bangladesh after independence
democratic_movements|subject|politics|ছাত্র আন্দোলন ও গণঅভ্যুত্থান|Student movements and mass uprisings|Student politics, popular protest, resistance to authoritarian rule and mass uprisings.|ছাত্র আন্দোলন;গণঅভ্যুত্থান;ঊনসত্তর;স্বৈরাচারবিরোধী;student movements;mass uprisings|ছাত্র আন্দোলন;গণঅভ্যুত্থান;ঊনসত্তর;স্বৈরাচার|ছাত্র আন্দোলন ও স্বৈরাচারবিরোধী সংগ্রাম নিয়ে বই|Books about student movements and resistance to authoritarian rule
july_uprising|subject|politics|জুলাই-আগস্ট ২০২৪ আন্দোলন|July-August 2024 movement|Catalogue accounts of the July-August 2024 protests, uprising and participants' experiences.|জুলাই অভ্যুত্থান;জুলাই বিপ্লব;জুলাই আগস্ট;কোটা আন্দোলন;July uprising;July August 2024|জুলাই;কোটা আন্দোলন|জুলাই-আগস্ট ২০২৪ আন্দোলনের প্রত্যক্ষ অভিজ্ঞতা নিয়ে বই|Books with firsthand accounts of the July-August 2024 movement
governance_democracy|subject|politics|গণতন্ত্র, সংবিধান ও শাসনব্যবস্থা|Democracy, constitutions and governance|Democratic institutions, elections, constitutional debates, public administration and state organisation.|গণতন্ত্র;সংবিধান;নির্বাচন;প্রশাসন;democracy;constitution;governance|গণতন্ত্র;সংবিধান;নির্বাচন;নির্বাচনি|বাংলাদেশের সংবিধান ও নির্বাচনি ব্যবস্থা নিয়ে বই|Books about Bangladesh's constitution and electoral system
international_relations|subject|politics|কূটনীতি ও আন্তর্জাতিক সম্পর্ক|Diplomacy and international relations|Foreign policy, diplomatic practice and relations among states.|কূটনীতি;পররাষ্ট্রনীতি;আন্তর্জাতিক সম্পর্ক;diplomacy;foreign policy;international relations|কূটনীতি;আন্তর্জাতিক সম্পর্ক;পররাষ্ট্রনীতি|বাংলাদেশের পররাষ্ট্রনীতি ও কূটনীতি নিয়ে বই|Books about Bangladesh's foreign policy and diplomacy
political_ideologies|subject|politics|রাজনৈতিক মতবাদ ও বিপ্লব|Political ideologies and revolutions|Marxism, socialism, capitalism, nationalism and revolutionary thought and movements.|মার্কসবাদ;সমাজতন্ত্র;পুঁজিবাদ;বিপ্লব;Marxism;socialism;capitalism;revolution|মার্কস;সমাজতন্ত্র;পুঁজিবাদ;বিপ্লব|পুঁজিবাদ ও সমাজতন্ত্রের তুলনা নিয়ে বই|Books comparing capitalism and socialism
social_anthropology|subject|society|সমাজবিজ্ঞান ও নৃবিজ্ঞান|Sociology and anthropology|Social structures, communities, cultural practices and anthropological studies.|সমাজবিজ্ঞান;নৃবিজ্ঞান;সমাজ নৃবিজ্ঞান;sociology;anthropology|সমাজবিজ্ঞান;নৃবিজ্ঞান|দক্ষিণ এশিয়ার সমাজ ও সংস্কৃতি নিয়ে নৃবিজ্ঞানের বই|Anthropological books about South Asian society and culture
folk_culture|subject|society|লোকসংস্কৃতি ও সাংস্কৃতিক ঐতিহ্য|Folklore and cultural heritage|Folk traditions, local customs, crafts, cultural identity and intangible heritage.|লোকসংস্কৃতি;ফোকলোর;লোকজ;হস্তশিল্প;folklore;folk culture;cultural heritage|লোকসংস্কৃতি;ফোকলোর;হস্তশিল্প|বাংলাদেশের লোকসংস্কৃতি ও লোকজ ঐতিহ্য নিয়ে বই|Books about Bangladeshi folklore and folk traditions
women_gender|subject|society|নারী, অধিকার ও সমাজ|Women, rights and society|Women's lives, social roles, education, rights and gender-related struggles.|নারী অধিকার;নারীমুক্তি;নারীবাদ;নারী শিক্ষা;women's rights;gender;feminism|নারী;নারীমুক্তি;রোকেয়া;রোকেয়া|নারী শিক্ষা ও সমাজে নারীর অবস্থান নিয়ে বই|Books about women's education and social position
ethnic_indigenous|subject|society|আদিবাসী ও জাতিগত জনগোষ্ঠী|Indigenous and ethnic communities|Indigenous communities, ethnic identities, minority experiences and the Chittagong Hill Tracts.|আদিবাসী;ক্ষুদ্র নৃগোষ্ঠী;পার্বত্য চট্টগ্রাম;indigenous peoples;ethnic communities|আদিবাসী;পার্বত্য চট্টগ্রাম;নৃগোষ্ঠী|পার্বত্য চট্টগ্রামের আদিবাসীদের ইতিহাস ও জীবন নিয়ে বই|Books about the history and lives of indigenous communities in the Hill Tracts
journalism|subject|society|সাংবাদিকতা ও গণমাধ্যম|Journalism and media|Newspapers, reporting, media history and journalists' professional experiences.|সাংবাদিকতা;সংবাদপত্র;গণমাধ্যম;পত্রিকা;journalism;newspapers;media history|সাংবাদিকতা;সাংবাদিকতায়;সংবাদপত্র;গণমাধ্যম|বাংলা সংবাদপত্র ও সাংবাদিকতার ইতিহাস নিয়ে বই|Books about the history of Bengali newspapers and journalism
islamic_history|subject|religion|ইসলামি ইতিহাস ও মুসলিম সভ্যতা|Islamic history and Muslim civilization|Muslim societies, Islamic dynasties, cultural achievements and historical developments.|ইসলামি ইতিহাস;মুসলিম সভ্যতা;ইসলামের ইতিহাস;Islamic history;Muslim civilization|ইসলামি ইতিহাস;ইসলামী ইতিহাস;মুসলিম সভ্যতা;ইসলামের ইতিহাস;মুসলিম শাসন|মুসলিম সভ্যতা ও ইসলামি ইতিহাস নিয়ে বই|Books about Muslim civilization and Islamic history
prophet_biography|subject|religion|নবীজীবন ও সিরাত|Prophets' lives and Sirah|Lives of prophets, particularly the life, character and mission of Prophet Muhammad.|সিরাত;সীরাত;নবীজীবন;মহানবী;Sirah;Seerah;life of Prophet Muhammad|সিরাত;সিরাতুন;সীরাত;মহানবী;নবি জীবনের|নবী মুহাম্মদের জীবন ও চরিত্র নিয়ে বই|Books about the life and character of Prophet Muhammad
companions_scholars|subject|religion|সাহাবি ও ইসলামি ব্যক্তিত্ব|Companions and Islamic figures|Lives and contributions of the Companions, caliphs, scholars and notable Islamic figures.|সাহাবি;সাহাবা;খলিফা;ইমাম;Companions;Sahaba;Islamic scholars|সাহাবি;সাহাবা;সাহাবীদের;সাহাবায়ে;সাহাবায়ে;খলিফা|সাহাবিদের জীবন ও জ্ঞানচর্চা নিয়ে বই|Books about the lives and scholarship of the Companions
islamic_teachings|subject|religion|কুরআন, হাদিস ও ইসলামি জীবন|Quran, Hadith and Islamic life|Islamic scripture, Hadith, ethics, faith and religious practice.|কুরআন;কোরআন;হাদিস;সুন্নাহ;ফিকহ;Quran;Hadith;Islamic ethics|কুরআন;হাদিস;সুন্নাহ;ঈমান;ফিকহুল|কুরআন ও হাদিসের আলোকে জীবন ও নৈতিকতা নিয়ে বই|Books about life and ethics in the Quran and Hadith
comparative_religion|subject|religion|ধর্মের ইতিহাস ও তুলনামূলক ধর্ম|Religious history and comparative religion|Histories, beliefs and traditions across religions, including Hinduism and Buddhism.|ধর্মের ইতিহাস;হিন্দুধর্ম;বৌদ্ধধর্ম;তুলনামূলক ধর্ম;religious history;comparative religion|ধর্মের ইতিহাস;হিন্দুধর্ম;বৌদ্ধধর্ম;ধর্মতত্ত্ব|হিন্দু ও বৌদ্ধ ধর্মের ইতিহাস সম্পর্কে বই|Books about the histories of Hinduism and Buddhism
sufism|subject|religion|সুফিবাদ ও মরমি ঐতিহ্য|Sufism and mystical traditions|Sufi thought, saints, shrines and mystical traditions in Bengal and elsewhere.|সুফিবাদ;সুফি;পির;দরবেশ;মরমি;Sufism;mystical traditions|সুফি;সুফিবাদ;সুফিয়ানা;মাজার সংস্কৃতি;পির দরবেশ|বাংলার সুফি সাধক ও মরমি ঐতিহ্য নিয়ে বই|Books about Sufi saints and mystical traditions in Bengal
language_literature|subject|literature|ভাষা, সাহিত্য ইতিহাস ও সমালোচনা|Language, literary history and criticism|Languages, scripts, literary development, interpretation and criticism of literary works.|ভাষা;সাহিত্য সমালোচনা;সাহিত্য ইতিহাস;লিপি;literary criticism;literary history;language studies|সাহিত্য;লিপি;চর্যাপদ|বাংলা সাহিত্যের ইতিহাস ও সাহিত্য সমালোচনা নিয়ে বই|Books about Bengali literary history and criticism
author_studies|subject|literature|রবীন্দ্রনাথ, নজরুল ও সাহিত্যিকদের জীবন-কর্ম|Writers' lives and works|Studies of writers and poets, their biographies, ideas, literary works and correspondence.|রবীন্দ্রনাথ;নজরুল;সাহিত্যিক;কবি;Tagore;Nazrul;author studies|রবীন্দ্রনাথ;নজরুল;মানিক বন্দ্যোপাধ্যায়;জীবনানন্দ|রবীন্দ্রনাথ ও নজরুলের জীবন ও সাহিত্যকর্ম নিয়ে বই|Books about Tagore's and Nazrul's lives and literary works
cinema|subject|arts|চলচ্চিত্র ও চলচ্চিত্র নির্মাণ|Cinema and filmmaking|Film history, criticism, filmmaking, directors and cinema culture.|চলচ্চিত্র;সিনেমা;চলচ্চিত্র নির্মাণ;ফিল্মমেকার;cinema;filmmaking;film criticism|চলচ্চিত্র;সিনেমা;ফিল্মমেকার|বাংলাদেশের চলচ্চিত্রের ইতিহাস ও নির্মাণশৈলী নিয়ে বই|Books about Bangladeshi cinema history and filmmaking
theatre|subject|arts|নাট্যকলা ও মঞ্চ|Theatre and performance|Theatre history, dramatic art, staging and performance traditions; use a form tag separately for play texts.|নাট্যকলা;মঞ্চ;থিয়েটার;নাট্য ইতিহাস;theatre;performing arts|নাটক ইতিহাস;থিয়েটার;থিয়েটার;নাট্য|নাট্যকলার ইতিহাস ও মঞ্চচর্চা নিয়ে বই|Books about theatre history and stage practice
music|subject|arts|সংগীত, গান ও সংগীতশিল্পী|Music, songs and musicians|Music history, folk and classical music, notation, song collections and musicians' lives.|সংগীত;সঙ্গীত;গান;লোকগান;স্বরবিতান;music;folk songs;musicians|সংগীত;সঙ্গীত;লোকগান;স্বরবিতান;গানের|বাংলার লোকগান ও সংগীতশিল্পীদের জীবন নিয়ে বই|Books about Bengali folk music and musicians' lives
visual_arts|subject|arts|চিত্রকলা, ভাস্কর্য ও আলোকচিত্র|Visual arts and photography|Painting, sculpture, art history, visual culture and photography.|চিত্রকলা;ভাস্কর্য;আলোকচিত্র;painting;sculpture;photography|চিত্রকলা;ভাস্কর্য;ভাস্কর্যে;আলোকচিত্রালাপ|বাংলাদেশের চিত্রকলা ও আলোকচিত্র নিয়ে বই|Books about Bangladeshi painting and photography
travel_writing|subject|travel|দেশ-বিদেশ ভ্রমণ|Travel and exploration|Journeys, destinations, observations and travel experiences in Bangladesh and abroad.|ভ্রমণ;ভ্রমণকাহিনি;ভ্রমণকাহিনী;পর্যটন;travel;travelogue;exploration|ভ্রমণ;ভ্রমণসমগ্র;ভ্রমণকাহিনি;ভ্রমণকাহিনী|দেশ-বিদেশের মানুষ ও সংস্কৃতি নিয়ে ভ্রমণকাহিনি|Travel books about people and cultures around the world
migration_diaspora|subject|travel|প্রবাস, অভিবাসন ও প্রবাসজীবন|Migration, diaspora and life abroad|Migration, expatriate lives, settlement abroad and diaspora experiences.|প্রবাস;প্রবাসী;অভিবাসন;প্রবাসজীবন;diaspora;migration;expatriate life|প্রবাস;প্রবাসী;প্রবাসে;অভিবাসন|প্রবাসে বাঙালিদের জীবন ও অভিজ্ঞতা নিয়ে বই|Books about Bengali diaspora experiences and life abroad
pilgrimage|subject|travel|হজ ও তীর্থভ্রমণ|Hajj and pilgrimage travel|Religious journeys, Hajj experiences and visits to pilgrimage destinations.|হজ;হজ্জ;তীর্থ;তীর্থভ্রমণ;Hajj;pilgrimage|হজ;হজ্জ;হজের;তীর্থ;হিন্দুতীর্থ|হজ ও ধর্মীয় তীর্থভ্রমণের অভিজ্ঞতা নিয়ে বই|Books about Hajj and religious pilgrimage experiences
economics_development|subject|business|অর্থনীতি ও উন্নয়ন|Economics and development|Economic theory, political economy, poverty, public finance and economic development.|অর্থনীতি;উন্নয়ন;দারিদ্র্য;রাজনৈতিক অর্থনীতি;economics;development;poverty|অর্থনীতি;অর্থনীতির;অর্থনৈতিক;দারিদ্র্য|বাংলাদেশের অর্থনীতি ও দারিদ্র্য নিয়ে বই|Books about Bangladesh's economy and poverty
islamic_finance|subject|business|ইসলামি অর্থনীতি ও ব্যবসায় নীতি|Islamic economics and business ethics|Islamic economic principles, interest-free finance, Islamic banking and religious business ethics.|ইসলামি অর্থনীতি;সুদমুক্ত;ইসলামী ব্যাংকিং;ইসলামি ব্যবসা;Islamic finance;Islamic economics|ইসলামি অর্থনীতি;ইসলামী অর্থনীতি;সুদমুক্ত;ইসলামী ব্যাংকিং;ফিকহুল বুয়ু|সুদমুক্ত অর্থায়ন ও ইসলামি ব্যবসায় নীতি নিয়ে বই|Books about interest-free finance and Islamic business ethics
banking_finance|subject|business|ব্যাংকিং, বীমা ও আর্থিক ব্যবস্থা|Banking, insurance and financial systems|Banks, banking practice, insurance, financial institutions and trade finance.|ব্যাংকিং;ব্যাংক;বীমা;আর্থিক ব্যবস্থা;banking;insurance;financial systems|ব্যাংকিং;ব্যাংক;বীমা;banking|ব্যাংকিং ও আন্তর্জাতিক বাণিজ্য অর্থায়ন নিয়ে বই|Books about banking and international trade finance
personal_finance|subject|business|ব্যক্তিগত অর্থব্যবস্থা ও সঞ্চয়|Personal finance and money management|Managing personal money, savings, budgeting and financial habits.|পারসোনাল ফাইন্যান্স;টাকা;সঞ্চয়;অর্থব্যবস্থাপনা;personal finance;money management|পারসোনাল ফাইন্যান্স;ফাইন্যান্স;মানি;money;টাকার|নিজের টাকা ও সঞ্চয় ভালোভাবে পরিচালনা করার বই|Books about managing personal money and savings
investment|subject|business|বিনিয়োগ ও শেয়ারবাজার|Investment and stock markets|Investment principles, shares, capital markets and market participation.|বিনিয়োগ;শেয়ারবাজার;শেয়ার;পুঁজিবাজার;investment;stock market|বিনিয়োগ;বিনিয়োগ;শেয়ার;শেয়ার;পুঁজিবাজার|শেয়ারবাজার ও বিনিয়োগের মৌলিক বিষয় নিয়ে বই|Books about the basics of investing and stock markets
marketing_sales|subject|business|মার্কেটিং, ব্র্যান্ডিং ও বিক্রয়|Marketing, branding and sales|Marketing strategy, brand building, selling, digital promotion and customer relationships.|মার্কেটিং;ব্র্যান্ডিং;বিক্রয়;ডিজিটাল মার্কেটিং;সেলস;marketing;branding;sales|মার্কেটিং;ব্র্যান্ডিং;বিক্রয়;বিক্রয়;marketing;সেলস|ডিজিটাল মার্কেটিং ও বিক্রয় বাড়ানোর কৌশল নিয়ে বই|Books about digital marketing and improving sales
entrepreneurship|subject|business|উদ্যোক্তা, ব্যবসা ও ই-কমার্স|Entrepreneurship, business and e-commerce|Starting and operating businesses, entrepreneurial thinking, online commerce and startups.|উদ্যোক্তা;ব্যবসা;ই-কমার্স;অনলাইন ব্যবসা;entrepreneurship;startup;e-commerce|উদ্যোক্তা;ব্যবসা;ব্যবসায়;ব্যবসায়;ই কমার্স|নতুন উদ্যোক্তাদের অনলাইন ব্যবসা শুরু করার বই|Books for new entrepreneurs starting an online business
management_leadership|subject|business|ব্যবস্থাপনা ও নেতৃত্ব|Management and leadership|Managing organisations and teams, leadership, strategy and organisational practice.|ব্যবস্থাপনা;ম্যানেজমেন্ট;নেতৃত্ব;লিডারশিপ;management;leadership|ম্যানেজমেন্ট;ব্যবস্থাপনা;লিডারশিপ;leadership;management|দল পরিচালনা ও নেতৃত্বের দক্ষতা বাড়ানোর বই|Books about team management and leadership skills
career_productivity|subject|business|ক্যারিয়ার, দক্ষতা ও ব্যক্তিগত উন্নতি|Career, skills and personal development|Professional growth, success habits, productivity, communication and customer service.|ক্যারিয়ার;সাফল্য;আত্মউন্নয়ন;কাস্টমার সার্ভিস;career;productivity;personal development|ক্যারিয়ার;ক্যারিয়ার;কাস্টমার সার্ভিস;হাউ টু উইন;সাকসেস|ক্যারিয়ার ও কাস্টমার সার্ভিসের দক্ষতা বাড়ানোর বই|Books about career growth and customer service skills
philosophy|subject|knowledge|দর্শন ও চিন্তাচর্চা|Philosophy and intellectual thought|Philosophical ideas, thinkers, ethics, reasoning and intellectual history.|দর্শন;দার্শনিক;চিন্তাচর্চা;philosophy;philosophers;intellectual history|দর্শন;দার্শনিক;দর্শনের|দর্শন ও দার্শনিকদের চিন্তা সম্পর্কে বই|Books about philosophy and philosophers' ideas
science|subject|knowledge|বিজ্ঞান ও বৈজ্ঞানিক চিন্তা|Science and scientific thought|Scientific ideas, discoveries, evolution and the lives and work of scientists.|বিজ্ঞান;বৈজ্ঞানিক;বিবর্তন;বিজ্ঞানী;science;scientific thought;evolution|বিজ্ঞান;বৈজ্ঞানিক;জৈববিবর্তনবাদ;বিজ্ঞানী|বিবর্তন ও বিজ্ঞানের ইতিহাস নিয়ে বই|Books about evolution and the history of science
education|subject|knowledge|শিক্ষা ও শিক্ষাপ্রতিষ্ঠান|Education and educational institutions|Educational thought, schools, universities, teaching and institutional histories.|শিক্ষা;শিক্ষাব্যবস্থা;বিদ্যালয়;বিশ্ববিদ্যালয়;education;universities;schools|শিক্ষা;শিক্ষাব্যবস্থা;বিশ্ববিদ্যালয়;বিশ্ববিদ্যালয়;স্কুল|বাংলাদেশের বিশ্ববিদ্যালয় ও শিক্ষাব্যবস্থার ইতিহাস নিয়ে বই|Books about the history of education and universities in Bangladesh
health_medicine|subject|knowledge|স্বাস্থ্য, চিকিৎসা ও জনস্বাস্থ্য|Health, medicine and public health|Medical experiences, healthcare, public health movements and medical figures.|স্বাস্থ্য;চিকিৎসা;জনস্বাস্থ্য;health;medicine;public health|জনস্বাস্থ্য;চিকিৎসা;স্বাস্থ্য অর্থনীতি|জনস্বাস্থ্য আন্দোলন ও চিকিৎসকদের অভিজ্ঞতা নিয়ে বই|Books about public health movements and doctors' experiences
sports|subject|knowledge|ক্রীড়া ও ক্রীড়াব্যক্তিত্ব|Sports and sporting figures|Sporting lives, athletes, sports personalities and sports culture.|ক্রীড়া;খেলাধুলা;ক্রিকেট;ফুটবল;sports;athletes|ক্রীড়া;ক্রীড়া;ক্রীড়াব্যাক্তিত্ব;ক্রীড়াঙ্গন;ক্রিকেট;ফুটবল|ক্রীড়াব্যক্তিত্বদের জীবন ও অভিজ্ঞতা নিয়ে বই|Books about the lives and experiences of sporting figures
history_form|form|formats|ইতিহাসগ্রন্থ|Historical accounts|Books presented as historical accounts; combine with a subject, place or period.|ইতিহাসগ্রন্থ;ঐতিহাসিক বিবরণ;history books;historical accounts|ইতিহাস|বাংলার ইতিহাসের বই|Historical accounts of Bengal
biography|form|formats|জীবনী|Biography|Accounts of another person's life, distinct from first-person autobiography.|জীবনী;জীবনচরিত;জীবন ও কর্ম;biography;life story|জীবনী;জীবনচরিত;জীবন ও কর্ম|একজন বিজ্ঞানীর জীবনী|A biography of a scientist
autobiography_memoir|form|formats|আত্মজীবনী ও স্মৃতিকথা|Autobiography and memoir|First-person life narratives, recollections and memories of particular experiences.|আত্মজীবনী;স্মৃতিকথা;স্মৃতিচারণ;আত্মকথা;autobiography;memoir|আত্মজীবনী;স্মৃতিকথা;স্মৃতিচারণ;আত্মকথা|একজন মুক্তিযোদ্ধার স্মৃতিকথা|A freedom fighter's memoir
diary|form|formats|ডায়েরি ও দিনলিপি|Diaries and journals|Dated journals, diaries and day-by-day records of personal or historical experience.|ডায়েরি;ডায়েরি;দিনলিপি;রোজনামচা;diary;journal|ডায়েরি;ডায়েরি;দিনলিপি;রোজনামচা|একাত্তরের যুদ্ধদিনের ডায়েরি|Diaries written during the 1971 war
novel|form|formats|উপন্যাস|Novels|Long-form fictional narratives; historical subject matter does not make them factual accounts.|উপন্যাস;ফিকশন;কিশোর উপন্যাস;novel;fiction|উপন্যাস|মুক্তিযুদ্ধের পটভূমিতে লেখা উপন্যাস|Novels set during the Liberation War
short_story|form|formats|গল্প ও ছোটগল্প|Stories and short fiction|Stories and short-story collections; narrative nonfiction can also use the word story, so verify the flap.|গল্প;ছোটগল্প;গল্পগ্রন্থ;short stories;story collection|ছোটগল্প;গল্প|দেশভাগ নিয়ে ছোটগল্প|Short stories about Partition
poetry|form|formats|কবিতা, কাব্য ও ছড়া|Poetry and rhymes|Poems, verse, poetry collections and rhymes.|কবিতা;কাব্য;ছড়া;ছড়া;poetry;verse;rhymes|কবিতা;কাব্য;ছড়া;ছড়া|স্বাধীনতা নিয়ে কবিতা ও ছড়া|Poetry and rhymes about independence
essay|form|formats|প্রবন্ধ ও নিবন্ধ|Essays and articles|Essays, analytical articles, columns and collected prose commentary.|প্রবন্ধ;নিবন্ধ;কলাম;essay;articles|প্রবন্ধ;নিবন্ধ;কলাম|সমাজ ও সংস্কৃতি নিয়ে প্রবন্ধ|Essays about society and culture
research|form|formats|গবেষণা ও বিশ্লেষণ|Research and analytical studies|Research-based studies, theses and analytical investigations of a topic.|গবেষণা;অভিসন্দর্ভ;বিশ্লেষণ;research;thesis;analytical study|গবেষণা;অভিসন্দর্ভ;বিশ্লেষণ|ভাষা আন্দোলন নিয়ে গবেষণাগ্রন্থ|Research studies about the Language Movement
reference|form|formats|অভিধান, তথ্যকোষ ও রেফারেন্স|Dictionaries, encyclopedias and reference|Works designed for consultation, including dictionaries, encyclopedias and glossaries.|অভিধান;তথ্যকোষ;বিশ্বকোষ;পরিভাষা;dictionary;encyclopedia;reference|অভিধান;তথ্যকোষ;বিশ্বকোষ;পরিভাষা|অর্থনীতির পরিভাষা শেখার অভিধান|A reference book explaining economic terminology
interview|form|formats|সাক্ষাৎকার|Interviews|Interviews, conversations and question-and-answer collections.|সাক্ষাৎকার;সাক্ষাত্কার;ইন্টারভিউ;কথোপকথন;interviews;conversations|সাক্ষাৎকার;সাক্ষাত্কার;ইন্টারভিউ|চলচ্চিত্র নির্মাতাদের সাক্ষাৎকার|Interviews with filmmakers
letters|form|formats|চিঠিপত্র|Letters and correspondence|Personal letters, correspondence and edited collections of letters.|চিঠিপত্র;চিঠি;পত্রাবলি;পত্রগুচ্ছ;letters;correspondence|চিঠিপত্র;চিঠি;পত্রাবলি;পত্রগুচ্ছ|রবীন্দ্রনাথের চিঠিপত্র|Collections of Tagore's letters
anthology|form|formats|সংকলন ও রচনাসমগ্র|Anthologies and collected works|Selected or collected writings, multi-author anthologies and complete works.|সংকলন;সমগ্র;রচনাবলি;নির্বাচিত;anthology;collected works|সংকলন;সমগ্র;রচনাবলি;রচনাবলী;নির্বাচিত|মুক্তিযুদ্ধের নির্বাচিত গল্পের সংকলন|An anthology of selected Liberation War stories
translation|form|formats|অনুবাদ|Translations|Works translated from another language; language of origin should be verified separately.|অনুবাদ;অনূদিত;ভাষান্তর;translation;translated books|অনুবাদ;অনূদিত;ভাষান্তর|বাংলায় অনূদিত বিশ্ব ইতিহাসের বই|World history books translated into Bengali
play|form|formats|নাটক|Plays|Dramatic texts and scripts; distinguish these from books analysing theatre.|নাটক;নাটিকা;নাট্যগ্রন্থ;play;drama;script|মুক্তিযুদ্ধের তিন নাটক|মুক্তিযুদ্ধ নিয়ে লেখা নাটক|Plays about the Liberation War
children_young_adults|audience|audiences|শিশু-কিশোর|Children and young adults|Books explicitly intended for children or young readers; combine with subject and form.|শিশু;কিশোর;ছোটদের;শিশুতোষ;children;young adults|ছোটদের;কিশোর;শিশু;শিশুতোষ|কিশোরদের জন্য মুক্তিযুদ্ধের গল্প|Liberation War stories for young readers
"""


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    # Treat punctuation/hyphens as word separators while preserving Bengali marks.
    return " ".join(re.sub(r"[^\w\u0980-\u09ff]+", " ", text).split())


def evidence_pattern(term: str) -> re.Pattern[str]:
    # Common Bengali inflections are allowed, but arbitrary substring matching is
    # avoided: ব্যাংক must not match ব্যাংকক, গান must not match গান্ধী.
    suffixes = "(?:ের|দের|গুলো|গুলি|তে|রা|র|ে|য়|য়)?"
    return re.compile(
        r"(?<![\w\u0980-\u09ff])"
        + re.escape(normalize(term))
        + suffixes
        + r"(?![\w\u0980-\u09ff])"
    )


def parse_definitions() -> list[dict]:
    categories = []
    for line in DEFINITIONS.strip().splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 10:
            raise ValueError(f"Expected 10 fields: {line}")
        cid, facet, group, bn, en, scope, aliases, terms, query_bn, query_en = fields
        categories.append({
            "id": cid,
            "facet": facet,
            "group_id": group,
            "label_bn": bn,
            "label_en": en,
            "description": scope,
            "aliases": aliases.split(";"),
            "evidence_terms": terms.split(";"),
            "example_queries": {"bn": query_bn, "en": query_en},
            "embedding_text": f"{bn}. {en}. {scope} Keywords: {aliases.replace(';', ', ')}.",
        })
    ids = [c["id"] for c in categories]
    assert len(ids) == len(set(ids)), "Category IDs must be unique"
    assert all(c["group_id"] in GROUPS for c in categories)
    assert all(c["facet"] in {"subject", "form", "audience"} for c in categories)
    return categories


def attach_evidence(categories: list[dict], rows: list[dict]) -> None:
    titles = [normalize(row["Book Name"]) for row in rows]
    flaps = None
    for category in categories:
        patterns = [evidence_pattern(term) for term in category["evidence_terms"]]
        hits = []
        for index, title in enumerate(titles):
            matches = [p.search(title) for p in patterns]
            lengths = [len(m.group()) for m in matches if m]
            if lengths:
                # Prefer specific evidence and shorter titles over huge book bundles.
                hits.append((max(lengths), -len(title), index, "Book Name", None))
        category["title_evidence_book_count"] = len(hits)
        if not hits:
            if flaps is None:
                flaps = [normalize(row["Description (Flap)"]) for row in rows]
            for index, flap in enumerate(flaps):
                for term, pattern in zip(category["evidence_terms"], patterns):
                    match = pattern.search(flap)
                    if match:
                        # Quote the original field, not the normalised match string.
                        original = rows[index]["Description (Flap)"]
                        original_pattern = re.compile(re.escape(term), re.IGNORECASE)
                        original_match = original_pattern.search(original)
                        if original_match:
                            start, end = original_match.span()
                            snippet = original[max(0, start - 80):min(len(original), end + 140)]
                            hits.append((len(match.group()), -len(titles[index]), index,
                                         "Description (Flap)", snippet))
                            break
        if not hits:
            raise ValueError(f"No catalogue evidence for {category['id']}")
        examples, seen_authors, seen_titles = [], set(), set()
        preferred = [normalize(t) for t in PREFERRED_TITLES.get(category["id"], [])]

        def rank(hit: tuple) -> tuple:
            title = titles[hit[2]]
            priority = preferred.index(title) if title in preferred else len(preferred)
            return priority, -hit[0], -hit[1], hit[2]

        ranked = sorted(hits, key=rank)
        for hit in ranked:
            index, field, snippet = hit[2:]
            row = rows[index]
            if row["Author"] in seen_authors or titles[index] in seen_titles:
                continue
            seen_authors.add(row["Author"])
            seen_titles.add(titles[index])
            example = {
                "source_record": index + 1,
                "title": row["Book Name"],
                "author": row["Author"],
                "evidence_field": field,
            }
            if snippet:
                example["evidence_excerpt"] = snippet
            examples.append(example)
            if len(examples) == 2:
                break
        category["representative_books"] = examples


def md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_exports(payload: dict, output_dir: Path, guide: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "book_categories.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = ["id", "facet", "group_id", "label_bn", "label_en", "description",
              "aliases", "example_query_bn", "example_query_en", "embedding_text",
              "title_evidence_book_count", "representative_books"]
    with (output_dir / "book_categories.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for category in payload["categories"]:
            record = {field: category.get(field, "") for field in fields}
            record["aliases"] = json.dumps(category["aliases"], ensure_ascii=False)
            record["representative_books"] = json.dumps(category["representative_books"], ensure_ascii=False)
            record["example_query_bn"] = category["example_queries"]["bn"]
            record["example_query_en"] = category["example_queries"]["en"]
            writer.writerow(record)
    categories = payload["categories"]
    counts = {facet: sum(c["facet"] == facet for c in categories)
              for facet in ("subject", "form", "audience")}
    lines = [
        "# Book categories for semantic queries", "",
        f"Source: `{payload['source']['file']}` — **{payload['source']['book_count']:,} books**.", "",
        f"**{len(categories)} categories:** {counts['subject']} subjects, "
        f"{counts['form']} book forms and {counts['audience']} audience category.", "",
        "This is a curated vocabulary grounded in catalogue titles and, where needed, "
        "book flaps. The source has no category column. Categories may overlap; "
        "the examples support their presence, but books have not been fully classified.", "",
        "## Using the list", "",
        "- Use `data/book_categories.json` for structured categories, bilingual labels, "
        "aliases, scope descriptions, query examples and source evidence.",
        "- Use `data/book_categories.csv` for spreadsheet review. List cells contain JSON arrays.",
        "- Embed each category's `embedding_text` using the same multilingual embedding "
        "model used for the query, then retrieve relevant category IDs as semantic hints.",
        "- Match a book using its title and flap. Keep its subject, form and audience "
        "as separate tags, and allow multiple tags per facet.",
        "- Retrieve and rank actual books using their metadata. Use category matches "
        "to assist retrieval; broad categories can miss distinctions within a subject.",
        "- Keep author, place, historical period and publication year as separate "
        "constraints. Publication year does not identify the era the book discusses.", "",
        "Example: **কিশোরদের জন্য নারী মুক্তিযোদ্ধাদের স্মৃতিকথা** combines "
        "`war_women` (subject), `autobiography_memoir` (form) and "
        "`children_young_adults` (audience). This illustrates query decomposition, "
        "not a guarantee that a book satisfying every constraint exists.", "",
        "The exports are reference data; the current search application does not load "
        "them automatically. The existing `data/taxonomy.yaml` is unchanged.", "",
    ]
    for group, (bn, en) in GROUPS.items():
        lines.extend([f"## {bn} / {en}", "",
                      "| Category ID | বাংলা / English | Scope | Example query | Catalogue evidence |",
                      "|---|---|---|---|---|"])
        for category in categories:
            if category["group_id"] != group:
                continue
            example = category["representative_books"][0]
            evidence = (f"{example['title']} — {example['author']} "
                        f"(record {example['source_record']})")
            if example["evidence_field"] != "Book Name":
                evidence += f"; flap: {example['evidence_excerpt']}"
            cells = [f"`{category['id']}`", f"{category['label_bn']} / {category['label_en']}",
                     category["description"], category["example_queries"]["bn"], evidence]
            lines.append("| " + " | ".join(md_cell(cell) for cell in cells) + " |")
        lines.append("")
    lines.extend([
        "## Rebuild and evidence notes", "",
        "Run `python scripts/build_book_categories.py` after catalogue changes. "
        "Edit the curated definitions in that script to change category scope or aliases.", "",
        "`source_record` is the 1-based data-record index (excluding the header), "
        "not a physical CSV line number or the search application's book ID. "
        "`title_evidence_book_count` counts literal evidence-term matches in titles; "
        "it is not a semantic category membership count. Flaps are checked only "
        "if a category has no title evidence. Author biographies are excluded "
        "because an author's interests do not establish a book's subject.", "",
        "The JSON records a SHA-256 fingerprint of the source file for reproducibility. "
        "Example queries illustrate intent; retrieval performance has not been evaluated.", "",
    ])
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "books_metadata_cleaned.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--guide", type=Path, default=ROOT / "BOOK_CATEGORIES.md")
    args = parser.parse_args()
    with args.source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Book Name", "Author", "Description (Flap)"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing required CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Source catalogue is empty")
    categories = parse_definitions()
    attach_evidence(categories, rows)
    fingerprint = hashlib.sha256()
    with args.source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    payload = {
        "schema_version": 1,
        "source": {"file": args.source.name, "book_count": len(rows),
                   "sha256": fingerprint.hexdigest(),
                   "fields_used": ["Book Name", "Description (Flap)"],
                   "record_numbering": "1-based data records, excluding the header"},
        "method": "Manually curated subjects, forms and audiences; source-matched examples. "
                  "Evidence matches are lexical hints, not complete book classifications.",
        "category_count": len(categories),
        "groups": [{"id": key, "label_bn": bn, "label_en": en}
                   for key, (bn, en) in GROUPS.items()],
        "categories": categories,
    }
    write_exports(payload, args.output_dir, args.guide)
    print(f"Exported {len(categories)} categories supported by {len(rows):,} source books.")
    for category in categories:
        example = category["representative_books"][0]
        print(f"{category['id']}: {category['title_evidence_book_count']} title evidence matches; "
              f"example: {example['title']} ({example['evidence_field']})")


if __name__ == "__main__":
    main()
