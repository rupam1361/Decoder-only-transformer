const dataset = [
  {
    context: "The capital of Japan is Tokyo .",
    question: "What is the capital of Japan?",
    answer: "Tokyo",
  },
  {
    context: "An octopus has three hearts and eight arms .",
    question: "How many hearts does an octopus have?",
    answer: "three",
  },
  {
    context: "The Amazon rainforest is the largest rainforest in the world .",
    question: "What is the largest rainforest in the world?",
    answer: "the Amazon rainforest",
  },
  {
    context: "The cheetah is known as the fastest land animal .",
    question: "What is the fastest land animal?",
    answer: "the cheetah",
  },
  {
    context: "The Great Wall of China was built to protect against invasions .",
    question: "Why was the Great Wall of China built?",
    answer: "to protect against invasions",
  },
  {
    context: "Honeybees produce honey from the nectar of flowers .",
    question: "What do honeybees make?",
    answer: "honey",
  },
  {
    context: "The Pacific Ocean is the largest ocean on Earth .",
    question: "Which is the largest ocean?",
    answer: "the Pacific Ocean",
  },
  {
    context: "Bananas are yellow when ripe and rich in potassium .",
    question: "What color are ripe bananas?",
    answer: "yellow",
  },
  {
    context: "The Earth orbits the Sun once every 365 days .",
    question: "How long does Earth take to orbit the Sun?",
    answer: "365 days",
  },
  {
    context: "The Sahara is the largest hot desert in the world .",
    question: "What is the largest hot desert?",
    answer: "the Sahara",
  },
  {
    context: "Cats often purr when they are happy or relaxed .",
    question: "When do cats often purr?",
    answer: "when they are happy or relaxed",
  },
  {
    context: "The Eiffel Tower is a famous landmark in Paris .",
    question: "Where is the Eiffel Tower located?",
    answer: "Paris",
  },
  {
    context: "Sharks have been living in the oceans for millions of years .",
    question: "Where do sharks live?",
    answer: "the oceans",
  },
  {
    context:
      "Lightning is a sudden discharge of electricity in the atmosphere .",
    question: "What is lightning a discharge of?",
    answer: "electricity",
  },
  {
    context: "Mount Everest is the tallest mountain in the world .",
    question: "What is the tallest mountain in the world?",
    answer: "Mount Everest",
  },
  {
    context: "Bamboo is the main food of the giant panda .",
    question: "What do giant pandas mainly eat?",
    answer: "bamboo",
  },
  {
    context: "The human heart pumps blood throughout the body .",
    question: "What does the heart pump?",
    answer: "blood",
  },
  {
    context: "Apples grow on trees and come in red, green, and yellow colors .",
    question: "Where do apples grow?",
    answer: "on trees",
  },
  {
    context: "Spiders spin webs to catch insects for food .",
    question: "Why do spiders spin webs?",
    answer: "to catch insects",
  },
  {
    context: "The Moon orbits the Earth and reflects sunlight .",
    question: "What does the Moon orbit?",
    answer: "the Earth",
  },
  {
    context: "Chocolate is made from cocoa beans .",
    question: "What is chocolate made from?",
    answer: "cocoa beans",
  },
  {
    context: "The camel is called the ship of the desert .",
    question: "Which animal is called the ship of the desert?",
    answer: "the camel",
  },
  {
    context: "Penguins cannot fly but are excellent swimmers .",
    question: "Can penguins fly?",
    answer: "no",
  },
  {
    context: "The human body has 206 bones .",
    question: "How many bones does the human body have?",
    answer: "206",
  },
  {
    context:
      "The Statue of Liberty was a gift from France to the United States .",
    question:
      "Which country gifted the Statue of Liberty to the United States?",
    answer: "France",
  },
  {
    context: "A year has 12 months .",
    question: "How many months are in a year?",
    answer: "12",
  },
  {
    context:
      "Rainbows appear when sunlight is refracted through water droplets .",
    question: "What causes rainbows to appear?",
    answer: "sunlight refracted through water droplets",
  },
  {
    context: "The koala is a marsupial found in Australia .",
    question: "Where are koalas found?",
    answer: "Australia",
  },
  {
    context: "Tigers have stripes that help them camouflage in forests .",
    question: "What helps tigers camouflage?",
    answer: "stripes",
  },
  {
    context: "The piano is a musical instrument with black and white keys .",
    question: "What color are the piano keys?",
    answer: "black and white",
  },
  {
    context: "The Nile River flows through Egypt and is very long .",
    question: "Which river flows through Egypt?",
    answer: "the Nile River",
  },
  {
    context: "Carrots are orange vegetables rich in vitamin A .",
    question: "What color are carrots?",
    answer: "orange",
  },
  {
    context: "The hummingbird can flap its wings extremely fast .",
    question: "Which bird can flap its wings very fast?",
    answer: "the hummingbird",
  },
  {
    context: "Salt is commonly used to season food .",
    question: "What is commonly used to season food?",
    answer: "salt",
  },
  {
    context: "The bicycle has two wheels and is powered by pedaling .",
    question: "How many wheels does a bicycle have?",
    answer: "two",
  },
  {
    context: "Whales are the largest animals that live in the ocean .",
    question: "What is the largest animal in the ocean?",
    answer: "whales",
  },
  {
    context: "Clouds are made of tiny water droplets or ice crystals .",
    question: "What are clouds made of?",
    answer: "water droplets or ice crystals",
  },
  {
    context: "The rose is a flower known for its pleasant fragrance .",
    question: "Which flower is known for its pleasant fragrance?",
    answer: "the rose",
  },
  {
    context: "Birds build nests to lay their eggs and protect them .",
    question: "Why do birds build nests?",
    answer: "to lay their eggs and protect them",
  },
  {
    context: "The computer uses electricity to function .",
    question: "What does a computer use to function?",
    answer: "electricity",
  },
  {
    context: "The lion is known as the king of the jungle .",
    question: "Which animal is known as the king of the jungle?",
    answer: "the lion",
  },
  {
    context: "Milk is a good source of calcium .",
    question: "What nutrient is milk rich in?",
    answer: "calcium",
  },
  {
    context: "The Earth is the third planet from the Sun .",
    question: "Which planet is Earth from the Sun?",
    answer: "third",
  },
  {
    context: "Snow is formed when water freezes in the clouds .",
    question: "Where is snow formed?",
    answer: "in the clouds",
  },
  {
    context: "The owl can see well at night .",
    question: "When can owls see well?",
    answer: "at night",
  },
  {
    context: "Bats use echolocation to navigate in the dark .",
    question: "What do bats use to navigate?",
    answer: "echolocation",
  },
  {
    context: "The orange is a citrus fruit rich in vitamin C .",
    question: "What vitamin is the orange rich in?",
    answer: "vitamin C",
  },
  {
    context: "Coins are small pieces of metal used as money .",
    question: "What are coins used as?",
    answer: "money",
  },
  {
    context: "The train runs on tracks and can travel long distances .",
    question: "What does a train run on?",
    answer: "tracks",
  },
  {
    context: "The zebra has black and white stripes on its body .",
    question: "What pattern does a zebra have?",
    answer: "black and white stripes",
  },
  {
    context: "The sunflower follows the direction of the sun .",
    question: "Which plant follows the direction of the sun?",
    answer: "the sunflower",
  },
  {
    context: "The mango is known as the king of fruits in India .",
    question: "Which fruit is known as the king of fruits?",
    answer: "the mango",
  },
  {
    context: "The dolphin is a friendly and intelligent sea animal .",
    question: "What type of animal is the dolphin?",
    answer: "a sea animal",
  },
  {
    context: "The refrigerator keeps food cold and fresh .",
    question: "What does a refrigerator do?",
    answer: "keeps food cold",
  },
  {
    context: "Gold is a yellow metal that is very valuable .",
    question: "What color is gold?",
    answer: "yellow",
  },
  {
    context: "The tomato is a red fruit often used in cooking .",
    question: "What color is a ripe tomato?",
    answer: "red",
  },
  {
    context: "A triangle has three sides .",
    question: "How many sides does a triangle have?",
    answer: "three",
  },
  {
    context: "The polar bear lives in the Arctic region .",
    question: "Where do polar bears live?",
    answer: "the Arctic",
  },
  {
    context: "The sunflower is bright yellow in color .",
    question: "What color is a sunflower?",
    answer: "yellow",
  },
  {
    context: "The kangaroo uses its strong legs to jump .",
    question: "What does a kangaroo use to jump?",
    answer: "strong legs",
  },
  {
    context: "Trees produce oxygen during photosynthesis .",
    question: "What do trees produce?",
    answer: "oxygen",
  },
  {
    context: "The Arctic Ocean is the smallest ocean .",
    question: "Which is the smallest ocean?",
    answer: "the Arctic Ocean",
  },
  {
    context: "The watermelon has a green rind and red flesh .",
    question: "What color is the inside of a watermelon?",
    answer: "red",
  },
  {
    context: "The library is a quiet place where people read books .",
    question: "What do people do in a library?",
    answer: "read books",
  },
  {
    context: "The eagle is known for its sharp eyesight .",
    question: "Which bird has sharp eyesight?",
    answer: "the eagle",
  },
  {
    context: "The rabbit has long ears and hops quickly .",
    question: "What feature do rabbits have?",
    answer: "long ears",
  },
  {
    context: "The clock on the wall shows the time .",
    question: "What does the clock show?",
    answer: "the time",
  },
  {
    context: "The pineapple has a spiky outer skin .",
    question: "What type of skin does a pineapple have?",
    answer: "spiky skin",
  },
  {
    context: "A rainbow usually has seven colors .",
    question: "How many colors does a rainbow have?",
    answer: "seven",
  },
  {
    context: "Lightning often occurs during thunderstorms .",
    question: "When does lightning usually occur?",
    answer: "during thunderstorms",
  },
  {
    context: "The pencil is used for writing and drawing .",
    question: "What is a pencil used for?",
    answer: "writing and drawing",
  },
  {
    context: "The polar regions are extremely cold .",
    question: "What is the temperature like in polar regions?",
    answer: "extremely cold",
  },
  {
    context: "The lemon tastes sour .",
    question: "What is the taste of a lemon?",
    answer: "sour",
  },
  {
    context: "The giraffe has a very long neck .",
    question: "What body part of a giraffe is long?",
    answer: "neck",
  },
  {
    context: "The onion has many layers inside .",
    question: "What does an onion have inside?",
    answer: "many layers",
  },
  {
    context: "Balloons are filled with air or helium .",
    question: "What are balloons filled with?",
    answer: "air or helium",
  },
  {
    context: "The sheep gives us wool .",
    question: "Which animal gives us wool?",
    answer: "the sheep",
  },
  {
    context: "The parrot can mimic human speech .",
    question: "Which bird can mimic speech?",
    answer: "the parrot",
  },
  {
    context: "Ice melts when it becomes warm .",
    question: "When does ice melt?",
    answer: "when it becomes warm",
  },
  {
    context: "The turtle carries a hard shell on its back .",
    question: "What does a turtle carry on its back?",
    answer: "a hard shell",
  },
  {
    context: "The fox is known for being clever .",
    question: "Which animal is known for being clever?",
    answer: "the fox",
  },
  {
    context: "The grapes grow in bunches .",
    question: "How do grapes grow?",
    answer: "in bunches",
  },
  {
    context: "The compass shows the direction .",
    question: "What does a compass show?",
    answer: "direction",
  },
  {
    context: "A mirror reflects light and images .",
    question: "What does a mirror reflect?",
    answer: "light and images",
  },
  {
    context: "The koala sleeps for most of the day .",
    question: "What do koalas do most of the day?",
    answer: "sleep",
  },
  {
    context: "The Earth has one moon .",
    question: "How many moons does Earth have?",
    answer: "one",
  },
  {
    context: "The flamingo is pink because of the food it eats .",
    question: "Why is the flamingo pink?",
    answer: "because of the food it eats",
  },
  {
    context: "The wolf lives and hunts in a pack .",
    question: "How does a wolf hunt?",
    answer: "in a pack",
  },
  {
    context: "A sandwich is made by placing food between slices of bread .",
    question: "What is used to make a sandwich?",
    answer: "slices of bread",
  },
  {
    context: "The ostrich is the largest bird in the world .",
    question: "What is the largest bird?",
    answer: "the ostrich",
  },
  {
    context: "The moon appears bright because it reflects sunlight .",
    question: "Why does the moon appear bright?",
    answer: "because it reflects sunlight",
  },
  {
    context: "The cricket chirps loudly at night .",
    question: "When do crickets chirp loudly?",
    answer: "at night",
  },
  {
    context: "A dictionary contains the meanings of words .",
    question: "What does a dictionary contain?",
    answer: "meanings of words",
  },
  {
    context: "The kangaroo carries its baby in a pouch .",
    question: "Where does a kangaroo carry its baby?",
    answer: "in a pouch",
  },
  {
    context: "The shark has very sharp teeth .",
    question: "What type of teeth do sharks have?",
    answer: "sharp teeth",
  },
  {
    context: "The camel can survive long without water .",
    question: "Which animal can survive long without water?",
    answer: "the camel",
  },
  {
    context: "The owl can rotate its head almost fully .",
    question: "Which animal can rotate its head a lot?",
    answer: "the owl",
  },
  {
    context: "The cricket match lasted for three hours .",
    question: "How long did the cricket match last?",
    answer: "three hours",
  },
  {
    context: "The baby cried because it was hungry .",
    question: "Why did the baby cry?",
    answer: "because it was hungry",
  },
  {
    context: "The apple pie smelled sweet and delicious .",
    question: "How did the apple pie smell?",
    answer: "sweet and delicious",
  },
  {
    context: "The laptop ran out of battery after many hours .",
    question: "Why did the laptop stop working?",
    answer: "it ran out of battery",
  },
  {
    context: "The bus stopped at the station to pick up the passengers .",
    question: "Why did the bus stop?",
    answer: "to pick up the passengers",
  },
  {
    context: "The teacher wrote the lesson on the board .",
    question: "Where did the teacher write the lesson?",
    answer: "on the board",
  },
  {
    context: "The boy drank water because he was thirsty .",
    question: "Why did the boy drink water?",
    answer: "because he was thirsty",
  },
  {
    context: "The phone vibrated because someone called .",
    question: "Why did the phone vibrate?",
    answer: "because someone called",
  },
  {
    context: "The rainbow appeared after the rain .",
    question: "When did the rainbow appear?",
    answer: "after the rain",
  },
  {
    context: "The wind blew strongly during the storm .",
    question: "When did the wind blow strongly?",
    answer: "during the storm",
  },
  {
    context: "The student studied hard for the exam .",
    question: "Why did the student study hard?",
    answer: "for the exam",
  },
  {
    context: "The gardener watered the plants daily .",
    question: "Who watered the plants?",
    answer: "the gardener",
  },
  {
    context: "The musician played a beautiful melody .",
    question: "What did the musician play?",
    answer: "a beautiful melody",
  },
  {
    context: "The baby slept peacefully in the cradle .",
    question: "Where did the baby sleep?",
    answer: "in the cradle",
  },
  {
    context: "The dog wagged its tail happily .",
    question: "What did the dog wag?",
    answer: "its tail",
  },
  {
    context: "The chef cooked a spicy meal .",
    question: "What type of meal did the chef cook?",
    answer: "a spicy meal",
  },
  {
    context: "The train arrived at the platform on time .",
    question: "Where did the train arrive?",
    answer: "the platform",
  },
  {
    context: "The children played games in the park .",
    question: "Where did the children play?",
    answer: "in the park",
  },
  {
    context: "The jacket kept him warm in winter .",
    question: "What kept him warm?",
    answer: "the jacket",
  },
  {
    context: "The teacher gave homework to the students .",
    question: "What did the teacher give?",
    answer: "homework",
  },
  {
    context: "The stars twinkled brightly in the night sky .",
    question: "Where did the stars twinkle?",
    answer: "in the night sky",
  },
  {
    context: "The painter used bright colors in the picture .",
    question: "What did the painter use?",
    answer: "bright colors",
  },
  {
    context: "The farmer grew rice in his field .",
    question: "What did the farmer grow?",
    answer: "rice",
  },
  {
    context: "The child smiled at the funny clown .",
    question: "Who did the child smile at?",
    answer: "the funny clown",
  },
  {
    context: "The frog jumped into the pond .",
    question: "Where did the frog jump?",
    answer: "into the pond",
  },
  {
    context: "The doctor checked the patient's temperature .",
    question: "What did the doctor check?",
    answer: "the patient's temperature",
  },
  {
    context: "The lion hunted its prey in the forest .",
    question: "Where did the lion hunt?",
    answer: "in the forest",
  },
  {
    context: "The chef baked a fresh loaf of bread .",
    question: "What did the chef bake?",
    answer: "loaf of bread",
  },
  {
    context: "The girl brushed her hair in the morning .",
    question: "What did the girl brush?",
    answer: "her hair",
  },
  {
    context: "The boy kicked the ball across the field .",
    question: "What did the boy kick?",
    answer: "the ball",
  },
  {
    context: "The snake slithered quietly through the grass .",
    question: "Where did the snake slither?",
    answer: "through the grass",
  },
  {
    context: "The monkey climbed the tall tree .",
    question: "What did the monkey climb?",
    answer: "the tall tree",
  },
  {
    context: "The teacher explained the lesson clearly .",
    question: "What did the teacher explain?",
    answer: "the lesson",
  },
  {
    context: "The artist drew a picture of the mountains .",
    question: "What did the artist draw?",
    answer: "a picture of the mountains",
  },
  {
    context: "The old man walked slowly with a stick .",
    question: "What did the old man walk with?",
    answer: "a stick",
  },
  {
    context: "The bus dropped the passengers at the station .",
    question: "Where did the bus drop the passengers?",
    answer: "at the station",
  },
  {
    context: "The bird built a nest on the tree .",
    question: "What did the bird build?",
    answer: "a nest",
  },
  {
    context: "The sun warmed the earth in the morning .",
    question: "What warmed the earth?",
    answer: "the sun",
  },
  {
    context:
      "The Solar System consists of the Sun and everything that orbits the Sun, including the eight major planets. The fourth planet from the Sun is Mars, often called the 'Red Planet' due to its reddish appearance. Mars is also the site of Olympus Mons, the largest volcano and highest known mountain in the Solar System.",
    question: "What is the largest mountain in the solar system?",
    answer: "Olympus Mons",
  },
];
