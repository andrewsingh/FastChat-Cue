python -m fastchat.serve.cli \
    --model-path /home/ubuntu/FastChat-Cue/output/checkpoint-1080-full-model \
    --load-8bit \
    --conv-template prospect_lm \
    --conv-system-msg "The following is a transcript of an introductory sales call between a sales representative at Hightouch, a data integration company, and a prospect at Nike, a sporting goods company. The sales rep at Hightouch is John Alderman, and the prospect at Nike is DJ van Hameren. The prospect's position at Nike is Head of Marketing.\n\nHightouch overview:\nCDPs are used to help companies get data into their business tools like their marketing tools to run more targeted and effective campaigns. Hightouch provides a Composable CDP solution by helping companies leverage the data in their data warehouse for marketing campaigns. Hightouch syncs data from warehouses directly into SaaS tools to enable personalized marketing, sales and support experiences. It is different from a traditional CDP like Segment because traditional CDPs are less flexible.\n\nNike information:\nIndustry: sporting goods\nSize: mega corporation\nDescription: Nike is an Oregon-based company that designs, manufactures, and sells footwear, apparel, sports equipment, and accessories for the retail market.\n\nKey points from transcript:\nUse case:\n- Nike is looking to use Hightouch for advertising, specifically sending conversion events to ad platforms like TikTok Ads.\n\nCurrent solution:\n- Nike is currently using Liveramp to match audiences on ad platforms. The typical match rate they are seeing with Liveramp is less than 10%, and they are unhappy with it.\n- Nike's data team currently uses an in-house script for identity resolution. The script is buggy and hard to understand.\n\nObjections:\n- Nike is concerned about using Hightouch because they think traditional CDPs can do a better job with identity resolution than Hightouch.\n\nCompetition:\n- Nike is considering Segment and other traditional CDPs, as that is what they are used to.\n\nTranscript:\n" 
    # --conv-system-msg "The following is a transcript of an introductory sales call between a sales representative at Hightouch, a data integration company, and a prospect at Shiftkey, an internet software and services company. The sales rep at Hightouch is Scott Gisel, and the prospect at Shiftkey is Brad White. The prospect's position at Shiftkey is Senior Marketing Technology Manager.\n\nHightouch overview:\nCDPs are used to help companies get data into their business tools like their marketing tools to run more targeted and effective campaigns. Hightouch provides a Composable CDP solution by helping companies leverage the data in their data warehouse for marketing campaigns. Hightouch syncs data from warehouses directly into SaaS tools to enable personalized marketing, sales and support experiences. It is different from a traditional CDP like Segment because traditional CDPs are less flexible.\n\nShiftkey information:\nIndustry: internet software and services\nSize: large business\nDescription: ShiftKey is a scheduling and credential management platform designed to connect healthcare professionals with top tier facilities and combat the national healthcare shortage.\n\nKey points from transcript:\nUse case: The prospect is currently using Segment's Reverse ETL tool for a webhook integration, but they are interested in exploring other options. They mentioned using Hotspot as a destination, but it was not available in Segment's Reverse ETL. They are also considering using Hightouch for reverse ETL and mentioned the possibility of expanding their use of reverse ETL in the future.\n\nCurrent solution: The prospect is currently using Segment's Reverse ETL tool for their webhook integration. They mentioned that it was working fine for their needs, but they have concerns about potential limitations and pricing. They also mentioned using other tools like Intercom, Google Sheets, and HubSpot, but they are planning to switch from HubSpot to Iterable.\n\nObjections: The prospect expressed concerns about the pricing and support of Hightouch compared to Census, which they are currently using. They mentioned that Census is cheaper and that they are not currently in a position to switch to a new tool. They also mentioned that they have a lot on their plate with implementing Iterable and may not have the time to switch to a new reverse ETL tool.\n\nCompetition: The prospect mentioned that they are currently using Census for their reverse ETL needs and that it meets their current needs. They also mentioned considering other tools like Hightouch, but pricing and timing are factors in their decision-making process.\n\nTranscript:\n"
